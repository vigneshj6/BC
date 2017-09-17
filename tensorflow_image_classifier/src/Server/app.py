import os
os.system("sudo pip install flask");

from flask import Flask, render_template, request
from werkzeug import secure_filename
import tensorflow as tf
import sys

app = Flask(__name__)


label_lines = [line.rstrip() for line in tf.gfile.GFile("/tf_files/retrained_labels.txt")]
f = tf.gfile.FastGFile("/tf_files/retrained_graph.pb", 'rb');
graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
_ = tf.import_graph_def(graph_def, name='')
sess = tf.Session()
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')


@app.route('/upload')
def upload_files():
   return render_template('upload.html')

@app.route('/uploader', methods = ['POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save('files/'+secure_filename(f.filename))
      print(f.filename);
      image_data = tf.gfile.FastGFile('files/'+secure_filename(f.filename), 'rb').read()
      predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
      top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
      result = [];
      for node_id in top_k:
          human_string = label_lines[node_id]
          score = predictions[0][node_id]
          result.append((human_string, score));
      print(str(result))
      return str(result);

if __name__ == '__main__':
   app.run('localhost',8888);
