import pandas as pd;
import random;
from preprocessing import toxic_comment_input_path, short_toxic_comment_input_path;



# gen a data file with num_toxic toxic comments and num_non_toxic non-toxic comments
def gen_dataset(num_toxic: int, num_non_toxic: int, data: pd.DataFrame) -> pd.DataFrame:
    toxic_index = random.sample(data[data['target'] > 0.5].index.values.tolist(), num_toxic);
    non_toxic_index = random.sample(data[data['target'] < 0.5].index.values.tolist(), num_non_toxic);
    return data.iloc[toxic_index + non_toxic_index, :];


if __name__ == '__main__':
    num_toxic = 5000;
    num_non_toxic = 5000;
    data = pd.read_csv(toxic_comment_input_path);
    data = gen_dataset(num_toxic, num_non_toxic, data);
    data.to_csv(short_toxic_comment_input_path);
