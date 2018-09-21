import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT

network_name = 'box_estimate'

num_class = 2

sample_num = 2048

batch_size = 12

num_epochs = 256

label_weights = []
for c in range(num_class):
    label_weights.append(1.0)

learning_rate_base = 0.001
decay_steps = 20000
decay_rate = 0.7
learning_rate_min = 1e-6

step_val = 500

weight_decay = 0.0

jitter = 0.0
jitter_val = 0.0

rotation_range = [0, math.pi/32., 0, 'u']
rotation_range_val = [0, 0, 0, 'u']
rotation_order = 'rxyz'

scaling_range = [0.0, 0.0, 0.0, 'g']
scaling_range_val = [0, 0, 0, 'u']

sample_num_variance = 1 // 8
sample_num_clip = 1 // 4

x = 4

xconv_param_name = ('K', 'D', 'P', 'C', 'links')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(12, 1, -1, 16 * x, []),
                 (16, 1, 768, 32 * x, []),
                 (16, 2, 384, 64 * x, []),
                 (16, 2, 128, 96 * x, [])]]

with_global = True

xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
xdconv_params = [dict(zip(xdconv_param_name, xdconv_param)) for xdconv_param in
                 [(16, 2, 3, 2),
                  (16, 1, 2, 1),
                  (12, 1, 1, 0)]]

fc_param_name = ('C', 'dropout_rate')
fc_params = []
# fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
#              [(512, 0.0),
#               (256, 0.0),
#               (3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, 0.0)]]

sampling = 'fps'

optimizer = 'adam'
epsilon = 1e-3

data_dim = 7
use_extra_features = True
with_normal_feature = False
with_X_transformation = True
sorting_method = None

keep_remainder = True
