# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Test MCore's metadata validation without executing TE kernels or collectives."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest import TestCase, mock

from megatron.core.extensions import transformer_engine as te
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType


class TestA2AHeadValidation(TestCase):
    """Keep the real MCore constructor/forward and mock only their TE boundary."""

    def setUp(self):
        if not te.HAVE_TE:
            self.skipTest("Transformer Engine is required to test the MCore wrapper")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.dict("os.environ", NVTE_APPLY_QK_LAYER_SCALING="0"))
        self.stack.enter_context(mock.patch.object(te, "is_te_min_version", return_value=True))
        self.stack.enter_context(
            mock.patch.object(te, "get_te_version", return_value=te.PkgVersion("2.10.0"))
        )
        self.stack.enter_context(
            mock.patch.object(
                te, "get_pg_size", side_effect=lambda pg: 1 if pg is None else pg.size()
            )
        )
        self.stack.enter_context(
            mock.patch.object(
                te.torch.distributed, "get_process_group_ranks", return_value=[0, 1, 2, 3]
            )
        )
        self.stack.enter_context(
            mock.patch.object(
                te, "get_cuda_rng_tracker", return_value=mock.Mock(is_initialized=lambda: False)
            )
        )
        self.stack.enter_context(
            mock.patch.object(te.TEDotProductAttention, "cp_stream", mock.sentinel.stream)
        )
        self.stack.enter_context(
            mock.patch.object(
                te,
                "get_hierarchical_context_parallel_groups",
                return_value=[self.group(2), self.group(4)],
            )
        )
        base = te.TEDotProductAttention.__bases__[0]

        def initialize(module, **kwargs):
            te.torch.nn.Module.__init__(module)
            module.cp_comm_type = kwargs.get("cp_comm_type", "p2p")
            module.cp_group = kwargs.get("cp_group")
            module.cp_global_ranks = kwargs.get("cp_global_ranks")

        def set_group(module, group, ranks, stream, comm):
            module.cp_group = group
            module.cp_global_ranks = ranks

        self.backend_init = self.stack.enter_context(
            mock.patch.object(base, "__init__", autospec=True, side_effect=initialize)
        )
        self.backend_forward = self.stack.enter_context(
            mock.patch.object(base, "forward", autospec=True, return_value=mock.sentinel.output)
        )
        self.backend_set_group = self.stack.enter_context(
            mock.patch.object(
                base, "set_context_parallel_group", autospec=True, side_effect=set_group
            )
        )

    @staticmethod
    def group(size):
        return mock.Mock(size=lambda: size)

    def attention(
        self, *, tp=1, cp=4, q=32, kv=4, comm="a2a", dynamic=False, mla=False, variant=None
    ):
        config = SimpleNamespace(
            tensor_model_parallel_size=tp,
            context_parallel_size=cp,
            num_attention_heads=q,
            num_query_groups=kv,
            kv_channels=128,
            dynamic_context_parallel=dynamic,
            multi_latent_attention=mla,
            experimental_attention_variant=variant,
            cp_comm_type=["p2p", "a2a"],
            batch_invariant_mode=False,
            apply_query_key_layer_scaling=False,
            deterministic_mode=False,
            window_size=None,
            window_attn_skip_freq=None,
            softmax_type="vanilla",
            qk_clip=False,
            log_max_attention_logit=False,
            attention_dropout=0.0,
            sequence_parallel=False,
        )
        groups = ProcessGroupCollection(tp=self.group(tp), cp=self.group(cp), hcp=[])
        return te.TEDotProductAttention(
            config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            cp_comm_type=comm,
            pg_collection=groups,
        )

    @staticmethod
    def forward(attention, heads=(32, 4, 4), packed=None, prefix=(16,)):
        tensors = [SimpleNamespace(shape=(*prefix, count, 128)) for count in heads]
        return attention.forward(*tensors, None, AttnMaskType.causal, packed_seq_params=packed)

    def test_static_invalid_cases_fail_before_te_constructor(self):
        for kwargs, q_heads, kv_heads, cp in (
            ({"cp": 8}, 32, 4, 8),
            ({"tp": 2, "cp": 4}, 16, 2, 4),
            ({"q": 6, "kv": 6, "cp": 4}, 6, 6, 4),
            ({"tp": 8, "kv": 4, "cp": 2}, 4, 1, 2),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError) as error:
                    self.attention(**kwargs)
                self.assertIn(f"local_q_heads={q_heads}", str(error.exception))
                self.assertIn(f"local_k_heads={kv_heads}", str(error.exception))
                self.assertIn(f"effective_a2a_size={cp}", str(error.exception))
                self.assertIn("explicitly select p2p", str(error.exception))
        self.backend_init.assert_not_called()

    def test_four_success_controls_reach_te(self):
        for kwargs, heads, runtime_size in (
            ({"cp": 4}, (32, 4, 4), None),
            ({"cp": 8, "comm": "p2p"}, (32, 4, 4), None),
            ({"tp": 2, "cp": 2}, (16, 2, 2), None),
            ({"cp": 4, "dynamic": True, "comm": "p2p"}, (32, 4, 4), 8),
        ):
            with self.subTest(kwargs=kwargs):
                attention = self.attention(**kwargs)
                packed = (
                    PackedSeqParams(local_cp_size=runtime_size, cp_group=self.group(runtime_size))
                    if runtime_size
                    else None
                )
                self.assertIs(self.forward(attention, heads, packed), mock.sentinel.output)
        self.assertEqual(self.backend_forward.call_count, 4)

    def test_runtime_invalid_group_does_not_mutate_te_state(self):
        for advertised_size in (4, 8):
            with self.subTest(advertised_size=advertised_size):
                attention = self.attention(cp=4, dynamic=True)
                original_group, original_ranks = attention.cp_group, attention.cp_global_ranks
                packed = PackedSeqParams(local_cp_size=advertised_size, cp_group=self.group(8))
                with self.assertRaisesRegex(
                    ValueError, "configured_cp_size=4, effective_a2a_size=8"
                ):
                    self.forward(attention, packed=packed)
                self.assertIs(attention.cp_group, original_group)
                self.assertIs(attention.cp_global_ranks, original_ranks)
        self.backend_set_group.assert_not_called()
        self.backend_forward.assert_not_called()

    def test_runtime_valid_shrink_and_disabled_cp(self):
        for runtime_size, heads in ((4, (32, 4, 4)), (1, (1, 1, 1))):
            with self.subTest(runtime_size=runtime_size):
                attention = self.attention(cp=8, dynamic=True)
                original_group = attention.cp_group
                packed = PackedSeqParams(local_cp_size=runtime_size, cp_group=self.group(4))
                self.assertIs(self.forward(attention, heads, packed), mock.sentinel.output)
                self.assertIs(attention.cp_group, original_group)
                active_group = self.backend_set_group.call_args_list[-2].args[1]
                self.assertIs(active_group, None if runtime_size == 1 else packed.cp_group)

    def test_runtime_checks_all_tensor_heads_and_layouts(self):
        for prefix in ((16,), (16, 1), (1, 16)):
            for heads in ((6, 4, 4), (32, 2, 4), (32, 4, 2)):
                with self.subTest(prefix=prefix, heads=heads):
                    attention = self.attention()
                    with self.assertRaises(ValueError):
                        self.forward(attention, heads, prefix=prefix)
        self.backend_forward.assert_not_called()
        self.backend_set_group.assert_not_called()

    def test_non_a2a_and_resolved_layer_comm_are_not_rejected(self):
        for comm in (None, "p2p", "all_gather", "a2a+p2p"):
            with self.subTest(comm=comm):
                attention = self.attention(cp=8, comm=comm)
                self.assertIs(self.forward(attention), mock.sentinel.output)
                self.assertEqual(attention.cp_comm_type, comm or "p2p")
        # Both layers share config.cp_comm_type=['p2p', 'a2a']; only the
        # resolved constructor argument selects which layer must be checked.
        with self.assertRaises(ValueError):
            self.attention(cp=8, comm="a2a")

    def test_mla_and_variants_defer_to_actual_tensors(self):
        for kwargs in ({"mla": True}, {"variant": "dsa"}):
            with self.subTest(kwargs=kwargs):
                attention = self.attention(cp=8, **kwargs)
                self.assertIs(self.forward(attention, (32, 32, 32)), mock.sentinel.output)
                with self.assertRaises(ValueError):
                    self.forward(attention, (32, 4, 4))

    def test_replicated_kv_is_one_and_cp1_is_valid(self):
        attention = self.attention(tp=8, kv=4, cp=1)
        self.assertIs(self.forward(attention, (4, 1, 1)), mock.sentinel.output)
        # Standard Attention normalizes the TE config to num_query_groups=TP.
        with self.assertRaisesRegex(ValueError, "local_k_heads=1, local_v_heads=1"):
            self.attention(tp=8, kv=8, cp=2)

    def test_dynamic_cp1_preserves_a2a_and_restores_disabled_state(self):
        attention = self.attention(cp=1, dynamic=True)
        self.assertEqual(attention.cp_comm_type, "a2a")
        original_group = attention.cp_group
        self.assertEqual(original_group.size(), 1)
        packed = PackedSeqParams(local_cp_size=4, cp_group=self.group(4))
        self.assertIs(self.forward(attention, packed=packed), mock.sentinel.output)
        self.assertIs(attention.cp_group, original_group)
        self.assertEqual(self.backend_set_group.call_count, 2)
        self.backend_set_group.reset_mock()
        self.backend_forward.reset_mock()
        packed = PackedSeqParams(local_cp_size=8, cp_group=self.group(8))
        with self.assertRaisesRegex(ValueError, "effective_a2a_size=8"):
            self.forward(attention, packed=packed)
        self.backend_set_group.assert_not_called()
        self.backend_forward.assert_not_called()

    def test_dynamic_group_is_required_before_state_change(self):
        attention = self.attention(dynamic=True)
        with self.assertRaisesRegex(AssertionError, "cp_group is not set"):
            self.forward(attention, packed=PackedSeqParams(local_cp_size=8))
        self.backend_set_group.assert_not_called()
        self.backend_forward.assert_not_called()
