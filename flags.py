import tensorflow as tf

def _print_default(define,name,default_value,descr):
    define(name,default_value,descr + " Default: " + str(default_value))

string_flag = lambda n,v,d: _print_default(tf.app.flags.DEFINE_string,n,v,d)
int_flag    = lambda n,v,d: _print_default(tf.app.flags.DEFINE_integer,n,v,d)
bool_flag   = lambda n,v,d: _print_default(tf.app.flags.DEFINE_boolean,n,v,d)
float_flag  = lambda n,v,d: _print_default(tf.app.flags.DEFINE_float,n,v,d)

FLAGS = tf.app.flags.FLAGS
