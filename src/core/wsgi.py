from app import flask_app as application
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if __name__ == '__main__':
    application.run(debug=False, host='0.0.0.0', port=7778)

