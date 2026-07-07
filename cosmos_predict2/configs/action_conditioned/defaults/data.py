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

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.action_conditioned.action_conditioned_dataset import ActionConditionedDataset
from imaginaire.lazy_config import LazyCall as L

base_path = "./datasets/bridge/"
train_annotation_path = os.path.join(base_path, "annotation/train")
val_annotation_path = os.path.join(base_path, "annotation/val")
test_annotation_path = os.path.join(base_path, "annotation/test")


bridge_train_dataset = L(ActionConditionedDataset)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=13,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="train",
)

bridge_val_dataset = L(ActionConditionedDataset)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=13,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="val",
)


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


bridge_train_dataloader = L(DataLoader)(
    dataset=bridge_train_dataset,
    sampler=L(get_sampler)(dataset=bridge_train_dataset),
    batch_size=1,
    drop_last=True,
)

bridge_val_dataloader = L(DataLoader)(
    dataset=bridge_val_dataset,
    sampler=L(get_sampler)(dataset=bridge_val_dataset),
    batch_size=1,
    drop_last=True,
)




# ===== touch-in-order (Luke): our own dataset, official ActionConditionedDataset format =====
import sys as _sys
_sys.path.insert(0, "/cephfs/shared/lyuxueguang/cosmos_touch")
from touch_action_dataset import TouchActionConditionedDataset as _TouchDS
touch_train_dataset = L(_TouchDS)(
    data_dir="/cephfs/shared/lyuxueguang/WAM_Experiments/runtime/touch_in_order_v7_isolated/data/v7_full/train",
    num_frames=13, video_size=(256, 256), max_shards=999,
)
touch_train_dataloader = L(DataLoader)(
    dataset=touch_train_dataset,
    sampler=L(get_sampler)(dataset=touch_train_dataset),
    batch_size=1, drop_last=True,
)

from touch_demoicl_dataset import TouchDemoICLActionDataset as _TouchICLDS
touch_icl_dataset = L(_TouchICLDS)(
    data_dir="/cephfs/shared/lyuxueguang/WAM_Experiments/runtime/touch_in_order_v7_isolated/data/v7_full/train",
    num_frames=13, td_demo=4, video_size=(256, 256), max_shards=999,
)
touch_icl_dataloader = L(DataLoader)(
    dataset=touch_icl_dataset,
    sampler=L(get_sampler)(dataset=touch_icl_dataset),
    batch_size=1, drop_last=True,
)


touch_icl_v8_dataset = L(_TouchICLDS)(
    data_dir="/cephfs/shared/lyuxueguang/WAM_Experiments/runtime/touch_in_order_v8_isolated/data/v8_full/train",
    num_frames=13, td_demo=4, video_size=(256, 256), max_shards=999,
)
touch_icl_v8_dataloader = L(DataLoader)(
    dataset=touch_icl_v8_dataset,
    sampler=L(get_sampler)(dataset=touch_icl_v8_dataset),
    batch_size=1, drop_last=True,
)


import sys as _sys
if '/cephfs/shared/lyuxueguang/cosmos_touch' not in _sys.path: _sys.path.insert(0,'/cephfs/shared/lyuxueguang/cosmos_touch')
from touch_coordtext_dataset import TouchCoordTextDataset as _TouchCTDS
touch_coordtext_v8_dataset = L(_TouchCTDS)(
    data_dir="/cephfs/shared/lyuxueguang/WAM_Experiments/runtime/touch_in_order_v8_isolated/data/v8_full/train", t5_cache="/cephfs/shared/lyuxueguang/WAM_Experiments/runtime/touch_in_order_v8_isolated/data/v8_full/t5_embeddings_coords.pkl",
    num_frames=13, video_size=(256, 256), max_shards=999,
)
touch_coordtext_v8_dataloader = L(DataLoader)(
    dataset=touch_coordtext_v8_dataset,
    sampler=L(get_sampler)(dataset=touch_coordtext_v8_dataset),
    batch_size=1, drop_last=True,
)


touch_coordtext_v7_dataset = L(_TouchCTDS)(
    data_dir="/cephfs/shared/lyuxueguang/WAM_Experiments/runtime/touch_in_order_v7_isolated/data/v7_full/train", t5_cache="/cephfs/shared/lyuxueguang/WAM_Experiments/runtime/touch_in_order_v7_isolated/data/v7_full/t5_embeddings_coords.pkl",
    num_frames=13, video_size=(256, 256), max_shards=999,
)
touch_coordtext_v7_dataloader = L(DataLoader)(
    dataset=touch_coordtext_v7_dataset,
    sampler=L(get_sampler)(dataset=touch_coordtext_v7_dataset),
    batch_size=1, drop_last=True,
)

def register_training_and_val_data_action_conditioned():
    cs = ConfigStore.instance()

    # for local dataset
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="bridge_train",
        node=bridge_train_dataloader,
    )
    cs.store(group="dataloader_train", package="dataloader_train", name="touch_train", node=touch_train_dataloader)
    cs.store(group="dataloader_train", package="dataloader_train", name="touch_icl_train", node=touch_icl_dataloader)
    cs.store(group="dataloader_train", package="dataloader_train", name="touch_icl_v8_train", node=touch_icl_v8_dataloader)
    cs.store(group="dataloader_train", package="dataloader_train", name="touch_coordtext_v8_train", node=touch_coordtext_v8_dataloader)
    cs.store(group="dataloader_train", package="dataloader_train", name="touch_coordtext_v7_train", node=touch_coordtext_v7_dataloader)
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="bridge_val",
        node=bridge_val_dataloader,
    )
