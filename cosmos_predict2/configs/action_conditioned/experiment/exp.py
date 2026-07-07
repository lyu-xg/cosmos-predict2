# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()

"""
torchrun --nproc_per_node=2 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment="predict2_video2world_2b_action_conditioned_training"
"""
predict2_video2world_2b_action_conditioned_training = dict(
    defaults=[
        {"override /model": "predict2_v2w_2b_action_conditioned_fsdp"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "bridge_train"},
        "_self_",
    ],
    model=dict(
        config=dict(
            fsdp_shard_size=-1,
        )
    ),
    job=dict(group="debug", name="predict2_video2world_2b_action_conditioned_training_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dict(
        batch_size=2,
    ),
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
)



predict2_video2world_2b_touch_action = dict(
    defaults=[
        {"override /model": "predict2_v2w_2b_action_conditioned_fsdp"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "touch_train"},
        "_self_",
    ],
    model=dict(config=dict(
        fsdp_shard_size=-1,
        model_manager_config=dict(dit_path="/cephfs/gyshare/lyuxueguang/cosmos_weights/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"),
    )),
    job=dict(group="cosmos_touch", name="predict2_video2world_2b_touch_action"),
    model_parallel=dict(context_parallel_size=1),
    dataloader_train=dict(batch_size=1),
    trainer=dict(distributed_parallelism="fsdp", max_iter=10000, logging_iter=25),
    checkpoint=dict(save_iter=500),
)


touch_cosmos_policy = dict(
    defaults=[
        {"override /model": "touch_cosmos_policy_fsdp_2b"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "touch_train"},
        "_self_",
    ],
    job=dict(group="cosmos_touch", name="touch_cosmos_policy"),
    model_parallel=dict(context_parallel_size=1),
    dataloader_train=dict(batch_size=1),
    trainer=dict(distributed_parallelism="fsdp", max_iter=10000, logging_iter=25),
    checkpoint=dict(save_iter=1000),
)


touch_cosmos_policy_icl = dict(
    defaults=[
        {"override /model": "touch_cosmos_policy_icl_fsdp_2b"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "touch_icl_train"},
        "_self_",
    ],
    job=dict(group="cosmos_touch", name="touch_cosmos_policy_icl"),
    model_parallel=dict(context_parallel_size=1),
    dataloader_train=dict(batch_size=1),
    trainer=dict(distributed_parallelism="fsdp", max_iter=10000, logging_iter=25),
    checkpoint=dict(save_iter=1000),
)


touch_cosmos_policy_icl_scratch = dict(
    defaults=[
        {"override /model": "touch_cosmos_policy_icl_fsdp_2b"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "touch_icl_train"},
        "_self_",
    ],
    model=dict(config=dict(model_manager_config=dict(dit_path="/cephfs/shared/lyuxueguang/cosmos_touch/random_init_2b.pt"))),  # SCRATCH: random-init ckpt (same load path as base)
    job=dict(group="cosmos_touch", name="touch_cosmos_policy_icl_scratch"),
    model_parallel=dict(context_parallel_size=1),
    dataloader_train=dict(batch_size=1),
    trainer=dict(distributed_parallelism="fsdp", max_iter=10000, logging_iter=25),
    checkpoint=dict(save_iter=1000),
)


touch_cosmos_policy_icl_v8 = dict(
    defaults=[
        {"override /model": "touch_cosmos_policy_icl_fsdp_2b"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "touch_icl_v8_train"},
        "_self_",
    ],
    job=dict(group="cosmos_touch", name="touch_cosmos_policy_icl_v8"),
    model_parallel=dict(context_parallel_size=1),
    dataloader_train=dict(batch_size=1),
    trainer=dict(distributed_parallelism="fsdp", max_iter=10000, logging_iter=25),
    checkpoint=dict(save_iter=1000),
)


touch_cosmos_policy_icl_scratch_v8 = dict(
    defaults=[
        {"override /model": "touch_cosmos_policy_icl_fsdp_2b"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "touch_icl_v8_train"},
        "_self_",
    ],
    model=dict(config=dict(model_manager_config=dict(dit_path="/cephfs/shared/lyuxueguang/cosmos_touch/random_init_2b.pt"))),
    job=dict(group="cosmos_touch", name="touch_cosmos_policy_icl_scratch_v8"),
    model_parallel=dict(context_parallel_size=1),
    dataloader_train=dict(batch_size=1),
    trainer=dict(distributed_parallelism="fsdp", max_iter=10000, logging_iter=25),
    checkpoint=dict(save_iter=1000),
)


touch_cosmos_policy_icl_cospolicy = dict(
    defaults=[
        {"override /model": "touch_cosmos_policy_icl_fsdp_2b"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "touch_icl_train"},
        "_self_",
    ],
    model=dict(config=dict(model_manager_config=dict(dit_path="/cephfs/gyshare/lyuxueguang/cosmos_policy_ckpts/Cosmos-Policy-LIBERO-Predict2-2B.pt"))),
    job=dict(group="cosmos_touch", name="touch_cosmos_policy_icl_cospolicy"),
    model_parallel=dict(context_parallel_size=1),
    dataloader_train=dict(batch_size=1),
    trainer=dict(distributed_parallelism="fsdp", max_iter=10000, logging_iter=25),
    checkpoint=dict(save_iter=1000),
)


touch_cosmos_policy_icl_cospolicy_v8 = dict(
    defaults=[
        {"override /model": "touch_cosmos_policy_icl_fsdp_2b"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "touch_icl_v8_train"},
        "_self_",
    ],
    model=dict(config=dict(model_manager_config=dict(dit_path="/cephfs/gyshare/lyuxueguang/cosmos_policy_ckpts/Cosmos-Policy-LIBERO-Predict2-2B.pt"))),
    job=dict(group="cosmos_touch", name="touch_cosmos_policy_icl_cospolicy_v8"),
    model_parallel=dict(context_parallel_size=1),
    dataloader_train=dict(batch_size=1),
    trainer=dict(distributed_parallelism="fsdp", max_iter=10000, logging_iter=25),
    checkpoint=dict(save_iter=1000),
)


touch_cosmos_policy_coordtext_cosmos2_v8 = dict(
    defaults=[
        {"override /model": "touch_cosmos_policy_fsdp_2b"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "touch_coordtext_v8_train"},
        "_self_",
    ],
    model=dict(config=dict(model_manager_config=dict(dit_path="/cephfs/gyshare/lyuxueguang/cosmos_weights/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"))),
    job=dict(group="cosmos_touch", name="touch_cosmos_policy_coordtext_cosmos2_v8"),
    model_parallel=dict(context_parallel_size=1),
    dataloader_train=dict(batch_size=1),
    trainer=dict(distributed_parallelism="fsdp", max_iter=10000, logging_iter=25),
    checkpoint=dict(save_iter=1000),
)


touch_cosmos_policy_coordtext_scratch_v8 = dict(
    defaults=[
        {"override /model": "touch_cosmos_policy_fsdp_2b"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "touch_coordtext_v8_train"},
        "_self_",
    ],
    model=dict(config=dict(model_manager_config=dict(dit_path="/cephfs/shared/lyuxueguang/cosmos_touch/random_init_2b.pt"))),
    job=dict(group="cosmos_touch", name="touch_cosmos_policy_coordtext_scratch_v8"),
    model_parallel=dict(context_parallel_size=1),
    dataloader_train=dict(batch_size=1),
    trainer=dict(distributed_parallelism="fsdp", max_iter=10000, logging_iter=25),
    checkpoint=dict(save_iter=1000),
)


touch_cosmos_policy_coordtext_cospolicy_v8 = dict(
    defaults=[
        {"override /model": "touch_cosmos_policy_fsdp_2b"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "touch_coordtext_v8_train"},
        "_self_",
    ],
    model=dict(config=dict(model_manager_config=dict(dit_path="/cephfs/gyshare/lyuxueguang/cosmos_policy_ckpts/Cosmos-Policy-LIBERO-Predict2-2B.pt"))),
    job=dict(group="cosmos_touch", name="touch_cosmos_policy_coordtext_cospolicy_v8"),
    model_parallel=dict(context_parallel_size=1),
    dataloader_train=dict(batch_size=1),
    trainer=dict(distributed_parallelism="fsdp", max_iter=10000, logging_iter=25),
    checkpoint=dict(save_iter=1000),
)


touch_cosmos_policy_coordtext_cosmos2_v7 = dict(
    defaults=[
        {"override /model": "touch_cosmos_policy_fsdp_2b"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "touch_coordtext_v7_train"},
        "_self_",
    ],
    model=dict(config=dict(model_manager_config=dict(dit_path="/cephfs/gyshare/lyuxueguang/cosmos_weights/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"))),
    job=dict(group="cosmos_touch", name="touch_cosmos_policy_coordtext_cosmos2_v7"),
    model_parallel=dict(context_parallel_size=1),
    dataloader_train=dict(batch_size=1),
    trainer=dict(distributed_parallelism="fsdp", max_iter=10000, logging_iter=25),
    checkpoint=dict(save_iter=1000),
)


touch_cosmos_policy_coordtext_scratch_v7 = dict(
    defaults=[
        {"override /model": "touch_cosmos_policy_fsdp_2b"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "touch_coordtext_v7_train"},
        "_self_",
    ],
    model=dict(config=dict(model_manager_config=dict(dit_path="/cephfs/shared/lyuxueguang/cosmos_touch/random_init_2b.pt"))),
    job=dict(group="cosmos_touch", name="touch_cosmos_policy_coordtext_scratch_v7"),
    model_parallel=dict(context_parallel_size=1),
    dataloader_train=dict(batch_size=1),
    trainer=dict(distributed_parallelism="fsdp", max_iter=10000, logging_iter=25),
    checkpoint=dict(save_iter=1000),
)


touch_cosmos_policy_coordtext_cospolicy_v7 = dict(
    defaults=[
        {"override /model": "touch_cosmos_policy_fsdp_2b"},
        {"override /optimizer": "adamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "touch_coordtext_v7_train"},
        "_self_",
    ],
    model=dict(config=dict(model_manager_config=dict(dit_path="/cephfs/gyshare/lyuxueguang/cosmos_policy_ckpts/Cosmos-Policy-LIBERO-Predict2-2B.pt"))),
    job=dict(group="cosmos_touch", name="touch_cosmos_policy_coordtext_cospolicy_v7"),
    model_parallel=dict(context_parallel_size=1),
    dataloader_train=dict(batch_size=1),
    trainer=dict(distributed_parallelism="fsdp", max_iter=10000, logging_iter=25),
    checkpoint=dict(save_iter=1000),
)

for _item in [
    # predict2_video2world_2b
    predict2_video2world_2b_action_conditioned_training,
    predict2_video2world_2b_touch_action,
    touch_cosmos_policy,
    touch_cosmos_policy_icl,
    touch_cosmos_policy_coordtext_cospolicy_v7,
    touch_cosmos_policy_coordtext_scratch_v7,
    touch_cosmos_policy_coordtext_cosmos2_v7,
    touch_cosmos_policy_coordtext_cospolicy_v8,
    touch_cosmos_policy_coordtext_scratch_v8,
    touch_cosmos_policy_coordtext_cosmos2_v8,
    touch_cosmos_policy_icl_cospolicy_v8,
    touch_cosmos_policy_icl_cospolicy,
    touch_cosmos_policy_icl_scratch_v8,
    touch_cosmos_policy_icl_v8,
    touch_cosmos_policy_icl_scratch,
]:
    # Get the experiment name from the global variable, e.g. exp01_wan_lora -> experiment_name = "exp01_wan_lora"
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
