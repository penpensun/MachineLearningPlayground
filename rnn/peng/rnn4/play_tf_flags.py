import tensorflow as tf;
#define flags
tf.flags.DEFINE_integer("age", 17, "age of user (default:20)");
tf.flags.DEFINE_boolean("drink_allow", False, "if can drink or not (default: False)");
tf.flags.DEFINE_float("weight", 55.55, "weight of user (default:55.55kg)");
tf.flags.DEFINE_string("name", "Lilei", "name of user (default:Lilei");

FLAGS = tf.flags.FLAGS;

#Get flags
for attr, value in FLAGS.__flags.items():
    print("attr:%s\tvalue:%s, value:%s" %(attr, str(value), str(FLAGS.attr) ) );

