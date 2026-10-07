# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Headwise CP metadata validation, without projections, FLA kernels, or collectives."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.gated_delta_net import common, gdn, kda
from megatron.core.ssm.gated_delta_net.common import _GDNBase


class _PastValidation(Exception):
    """Stop before tensor computation or parameter allocation."""


class _InputShapeBoundary:
    @property
    def shape(self):
        raise _PastValidation


def _group(size):
    return SimpleNamespace(size=lambda: size)


def _metadata(key_heads=4, value_heads=8, configured_cp=4, mode="headwise", dynamic=False):
    config = SimpleNamespace(
        linear_cp_mode=mode,
        cp_comm_type="p2p",  # Softmax communication does not control GDN headwise CP.
        context_parallel_size=configured_cp,
        dynamic_context_parallel=dynamic,
        tensor_model_parallel_size=2,
        sequence_parallel=False,
        hidden_size=128,
        activation_func=lambda x: x,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=key_heads * 2,
        linear_num_value_heads=value_heads * 2,
    )
    module = SimpleNamespace(
        config=config,
        tp_size=2,
        num_k_heads_local_tp=key_heads,
        num_v_heads_local_tp=value_heads,
        pg_collection=SimpleNamespace(tp=_group(2), cp=_group(configured_cp)),
    )
    module.tp_group = module.pg_collection.tp
    module._validate_headwise_cp = MethodType(_GDNBase._validate_headwise_cp, module)
    return module


def _check_error(error, module, effective_cp):
    for detail in (
        "linear_cp_mode='headwise'",
        "TP=2",
        f"local_key_heads={module.num_k_heads_local_tp}",
        f"local_value_heads={module.num_v_heads_local_tp}",
        f"configured_cp_size={module.config.context_parallel_size}",
        f"effective_cp_size={effective_cp}",
        "chunkwise",
    ):
        assert detail in str(error)


@pytest.mark.parametrize("key_heads,value_heads", [(2, 8), (8, 2), (6, 12)])
def test_headwise_rejects_each_nondivisible_local_head_count(key_heads, value_heads):
    module = _metadata(key_heads, value_heads)
    with pytest.raises(ValueError) as error:
        module._validate_headwise_cp(4)
    _check_error(error.value, module, 4)


@pytest.mark.parametrize(
    "key_heads,value_heads,effective_cp,mode",
    [(4, 8, 4, "headwise"), (1, 1, 1, "headwise"), (2, 3, 8, "chunkwise")],
)
def test_valid_head_metadata_is_accepted(key_heads, value_heads, effective_cp, mode):
    module = _metadata(key_heads, value_heads, mode=mode)
    module._validate_headwise_cp(effective_cp)


@pytest.mark.parametrize(
    "configured_cp,mode,dynamic,rejected",
    [
        (4, "headwise", False, False),
        (8, "headwise", False, True),
        (8, "headwise", True, False),
        (8, "chunkwise", False, False),
    ],
)
def test_constructor_validates_before_variant_setup(
    monkeypatch, configured_cp, mode, dynamic, rejected
):
    module = _metadata(configured_cp=configured_cp, mode=mode, dynamic=dynamic)
    monkeypatch.setattr(common, "HAVE_FLA", True)
    setup = Mock(side_effect=_PastValidation)
    monkeypatch.setattr(_GDNBase, "_setup_variant_attrs", setup)
    with pytest.raises(ValueError if rejected else _PastValidation) as error:
        _GDNBase(module.config, submodules=None, pg_collection=module.pg_collection)
    if rejected:
        _check_error(error.value, module, configured_cp)
        setup.assert_not_called()
    else:
        setup.assert_called_once()


@pytest.mark.parametrize("variant", [gdn.GatedDeltaNet, kda.KimiDeltaAttention])
@pytest.mark.parametrize(
    "configured_cp,effective_cp,source,mode,key_heads,rejected",
    [
        (8, 4, "packed", "headwise", 4, False),
        (4, 8, "packed", "headwise", 4, True),
        (8, 4, "override", "headwise", 4, False),
        (4, 8, "override", "headwise", 4, True),
        (8, 1, "packed", "headwise", 4, False),
        (4, 8, "packed", "chunkwise", 4, False),
        # The 64 local key channels divide CP4, but the two local heads do not.
        (4, 4, "packed", "headwise", 2, True),
    ],
)
def test_forward_uses_resolved_group_before_tensor_work(
    monkeypatch, variant, configured_cp, effective_cp, source, mode, key_heads, rejected
):
    module = _metadata(
        key_heads=key_heads,
        value_heads=key_heads if variant is kda.KimiDeltaAttention else 8,
        configured_cp=configured_cp,
        mode=mode,
        dynamic=True,
    )
    effective_group = _group(effective_cp)
    packed = PackedSeqParams(cp_group=effective_group) if source == "packed" else None
    override = SimpleNamespace(cp=effective_group) if source == "override" else None
    variant_module = gdn if variant is gdn.GatedDeltaNet else kda
    # Chunkwise layout conversion is outside this metadata test's scope.
    monkeypatch.setattr(
        variant_module,
        "convert_module_input_tensors_cp_partition_mode",
        lambda **kwargs: (kwargs["hidden_states"], None),
    )
    with pytest.raises(ValueError if rejected else _PastValidation) as error:
        variant.forward(
            module,
            _InputShapeBoundary(),
            attention_mask=None,
            packed_seq_params=packed,
            pg_collection=override,
        )
    if rejected:
        _check_error(error.value, module, effective_cp)
