import os
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf_flags = tf.app.flags

tf_flags.DEFINE_string('logdir', 'log/test', 'Directory where to write event logs.')
tf_flags.DEFINE_string('train_data', '', 'Training data.')
tf_flags.DEFINE_string('test_data', '', 'Test data.')
tf_flags.DEFINE_string('test_data_visual', '', 'Testing data for visualization.')
tf_flags.DEFINE_integer('train_batch_size', 8, 'Batch size for the training.')
tf_flags.DEFINE_integer('test_batch_size', 1, 'Batch size for the testing.')
tf_flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
tf_flags.DEFINE_string('optimizer', 'sgd', 'Optimizer (adam/sgd).')
tf_flags.DEFINE_string('decay_policy', 'step', 'Learning rate decay policy (step/poly/constant).')
tf_flags.DEFINE_float('weight_decay', 0.0001, 'Weight decay.')
tf_flags.DEFINE_integer('max_iter', 100000, 'Maximum training iterations.')
tf_flags.DEFINE_integer('test_every_iter', 5000, 'Test model every n training steps.')
tf_flags.DEFINE_integer('test_iter', 100, '#shapes in test data.')
tf_flags.DEFINE_integer('test_iter_visual', 20, 'Test steps in testing phase for visualization.')
tf_flags.DEFINE_boolean('test_visual', False, """Test with visualization.""")
tf_flags.DEFINE_string('cache_folder', 'test', 'Directory where to dump immediate data.')
tf_flags.DEFINE_string('ckpt', '', 'Restore weights from checkpoint file.')
tf_flags.DEFINE_string('gpu', '0', 'The gpu index.')
tf_flags.DEFINE_string('phase', 'train', 'Choose from train, test or dump}.')
tf_flags.DEFINE_integer('n_part_1', 6, 'Number of semantic part in level one.')
tf_flags.DEFINE_integer('n_part_2', 30, 'Number of semantic part in level two.')
tf_flags.DEFINE_integer('n_part_3', 39, 'Number of semantic part in level three.')
tf_flags.DEFINE_boolean('delete_0', True, """Whether consider label 0 in metric computation.""")
tf_flags.DEFINE_float('seg_loss_weight', 1.0, 'Weight of segmentation loss.')
tf_flags.DEFINE_float('offset_weight', 1.0, 'Weight of offset loss.')
tf_flags.DEFINE_float('sem_offset_weight', 1.0, 'Weight of semantic offset loss.')
tf_flags.DEFINE_float('level_1_weight', 0.0, 'Weight of level 1 loss.')
tf_flags.DEFINE_float('level_2_weight', 0.0, 'Weight of level 2 loss.')
tf_flags.DEFINE_float('level_3_weight', 0.0, 'Weight of level 3 loss.')
tf_flags.DEFINE_integer('test_shape_average_point_number', 10000, 'Mean point number of each shape in test phase. Must be greater than real point number.')
tf_flags.DEFINE_integer('depth', 6, 'The depth of octree.')
tf_flags.DEFINE_boolean('stop_gradient', True, """Stop gradient in fusion module.""")
tf_flags.DEFINE_string('category', 'Chair', 'Category.')
tf_flags.DEFINE_float('bandwidth', 0.1, 'Bandwidth of mean-shift.')
tf_flags.DEFINE_float('semantic_center_offset', 0.05, 'semantic center offset.')


FLAGS = tf_flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

max_iter = FLAGS.max_iter
test_iter = FLAGS.test_iter
test_iter_visual = FLAGS.test_iter_visual
n_part_1 = FLAGS.n_part_1
n_part_2 = FLAGS.n_part_2
n_part_3 = FLAGS.n_part_3
n_test_point = FLAGS.test_shape_average_point_number
