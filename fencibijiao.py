from bert_serving.client import BertClient
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy import stats
import numpy as np

class Encoding(object):
    def __init__(self):
        self.server_ip = "localhost"
        self.bert_client = BertClient(ip=self.server_ip)

    def encode(self, query):
        tensor = self.bert_client.encode([query])
        return tensor

    def query_similarity(self, query_list):
        tensors = self.bert_client.encode(query_list)
        # dist = np.linalg.norm(tensors[0]-tensors[1])
        # prea = stats.pearsonr(tensors[0],tensors[1])[0]
        return cosine_similarity(tensors)[0][1]
    # 在数字和汉字上似乎欧氏距离和scipy包的余弦相似度泛化性更好
    # 再具体的汉语中，皮埃尔系数似乎更好（sklearn的余弦包和皮埃尔一致）
    #第一次尝试选择skleran
def Encode(wv_tok,nlu_t1):
    vec = Encoding()
    max_sum = [0, 0, 0, 0]
    idx_sum = [0, 0, 0, 0]

    for size in range(1,5):
        similarity_max1 = 0

        if nlu_t1 is None or size > len(nlu_t1) or size == 0:
            print("滑动窗口出错")
            print(nlu_t1)
            print(size)
            break
        for i in range(len(nlu_t1) - size + 1):

            size_b1 = ''.join(nlu_t1[i:i + size]) #滑动size加入列表 i是起始索引，i+size是结束索引
            similarity = vec.query_similarity([size_b1, wv_tok])
            if similarity > similarity_max1:
                similarity_max1 = similarity
                max_sum[size-1] = similarity_max1
                idx_sum[size-1] = [i, i+size]
            else:
                continue

    a = max(max_sum)
    for idx, val in enumerate(max_sum):
        if val == a:
            suoyin = idx_sum[idx]
    suoyin = [suoyin[0], suoyin[1]-1] #ed索引就是实际位置
    return suoyin

