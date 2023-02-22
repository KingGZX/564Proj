from helper_s3dis import Data_S3DIS
import glob

a = 'Area_1'
con = glob.glob("./data/Data_S3DIS/" + a + '*.h5')

data = Data_S3DIS("./data/Data_S3DIS/", ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6'], ['Area_5'])

data.load_train_next_batch()
