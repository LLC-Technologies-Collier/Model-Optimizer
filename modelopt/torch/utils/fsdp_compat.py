# --- c9h Global Mock Block ---
import torch
import torch.nn as nn
try:
    from torch.distributed.fsdp import FSDPModule
except ImportError:
    class FSDPModule: pass

try:
    from torch.distributed.fsdp import MixedPrecisionPolicy
except ImportError:
    try:
        from torch.distributed.fsdp import MixedPrecision as MixedPrecisionPolicy
    except ImportError:
        class MixedPrecisionPolicy: pass

try:
    from torch.distributed.fsdp import fully_shard
except ImportError:
    def fully_shard(module, **kwargs): return module

try:
    from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
except ImportError:
    class FSDPParam: pass

try:
    from torch.distributed.tensor import DTensor, Replicate, DeviceMesh, Shard
except ImportError:
    try:
        from torch.distributed._tensor import DTensor, Replicate, DeviceMesh, Shard
    except ImportError:
        class DTensor: pass
        class Replicate: pass
        class DeviceMesh: pass
        class Shard: pass
# -----------------------------

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
import torch.nn as nn