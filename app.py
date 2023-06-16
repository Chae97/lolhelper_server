from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import random
import json
import numpy as np

app = Flask(__name__)



# server init
base_uri = "/ai/dev/team2"

top_model_path = "./model/linear_top_predictor_v4.h5"
jg_model_path = "./model/linear_jg_predictor_v4.h5"
mid_model_path = "./model/linear_mid_predictor_v4.h5"
ad_model_path = "./model/linear_ad_predictor_v4.h5"
sup_model_path = "./model/linear_sup_predictor_v4.h5"

top_model = tf.keras.models.load_model(top_model_path)
jg_model = tf.keras.models.load_model(jg_model_path)
mid_model = tf.keras.models.load_model(mid_model_path)
ad_model = tf.keras.models.load_model(ad_model_path)
sup_model = tf.keras.models.load_model(sup_model_path)

top_list = []
jg_list = []
mid_list = []
ad_list = []
sup_list = []

with open("./assets/top_list.json", 'r', encoding="utf-8") as f:
    top_list = json.load(f)
f.close()

with open("./assets/jg_list.json", 'r', encoding="utf-8") as f:
    jg_list = json.load(f)
f.close()

with open("./assets/mid_list.json", 'r', encoding="utf-8") as f:
    mid_list = json.load(f)
f.close()

with open("./assets/ad_list.json", 'r', encoding="utf-8") as f:
    ad_list = json.load(f)
f.close()

with open("./assets/sup_list.json", 'r', encoding="utf-8") as f:
    sup_list = json.load(f)
f.close()

all_lane = (top_list, jg_list, mid_list , ad_list, sup_list)



@app.route(base_uri + "/endgame", methods=['GET', 'POST'])
def endgame():
    if request.method =='POST':
        endgame_json = request.get_json()
        print(endgame_json)
        return "sdf"



@app.route(base_uri + "/predict", methods=['GET', 'POST'])
def predict():
    if request.method =='POST':
        client_json = request.get_json()


        user_most = client_json["most_champion"]
        user_lane = client_json["line"]
        ban_state = client_json["ban"]
        possible_champion = client_json["possible_champion"]

        get_winning_rate_by_champion(user_lane,user_most)
        get_softmax_winning_rate_list(user_lane,user_most)
        weight_list = weight_caculate(user_most)

        input_data = preprocess_input_data(client_json)
        # print("input_data", input_data)
        pred_result = model_predict(input_data, user_lane)
        result = preprocess_pred(pred_result, ban_state, user_lane, user_most, input_data, weight_list, possible_champion)
        return jsonify(result)
        
@app.route(base_uri + "/test")
def test():

    # client_json = request.get_json()
    client_json = {
        'line': 4, #// top: 1, jungle: 2, mid: 3, bottom: 4, utility: 5
        'myTeam': {
            'top': 0,
            'jungle': 0,
            'middle': 51,
            'bottom': 0,
            'utility': 221
        },
        'theirTeam': [202, 0, 145, 0, 30],
        'ban': [96, 236, 21, 895, 360, 147, 15, 18, 29, 110],
        'most_champion' : [{'id': 498, 'games': 14, 'winrate': 42}, {'id': 81, 'games': 12, 'winrate': 75}, {'id': 221, 'games': 10, 'winrate': 40}, {'id': 523, 'games': 8, 'winrate': 62}, {'id': 24, 'games': 4, 'winrate': 25}, {'id': 145, 'games': 4, 'winrate': 75}, {'id': 222, 'games': 4, 'winrate': 25}, {'id': 15, 'games': 3, 'winrate': 66}, {'id': 90, 'games': 3, 'winrate': 33}, {'id': 85, 'games': 3, 'winrate': 33}, {'id': 236, 'games': 3, 'winrate': 33}, {'id': 7, 'games': 3, 'winrate': 33}, {'id': 202, 'games': 2, 'winrate': 100}, {'id': 777, 'games': 2, 'winrate': 50}, {'id': 114, 'games': 2, 'winrate': 50}, {'id': 1, 'games': 2, 'winrate': 100}, {'id': 98, 'games': 1, 'winrate': 0}, {'id': 8, 'games': 1, 'winrate': 0}, {'id': 61, 'games': 1, 'winrate': 0}, {'id': 235, 'games': 1, 'winrate': 100}, {'id': 34, 'games': 1, 'winrate': 0}, {'id': 43, 'games': 1, 'winrate': 100}, {'id': 157, 'games': 1, 'winrate': 0}],
        'possible_champion' : [81, 157, 498]
    }


    
    user_most = client_json["most_champion"]
    user_lane = client_json["line"]
    ban_state = client_json["ban"]
    possible_champion = client_json["possible_champion"]

    get_winning_rate_by_champion(user_lane,user_most)
    get_softmax_winning_rate_list(user_lane,user_most)
    weight_list = weight_caculate(user_most)

    input_data = preprocess_input_data(client_json)
    # print("input_data", input_data)
    pred_result = model_predict(input_data, user_lane)
    result = preprocess_pred(pred_result, ban_state, user_lane, user_most, input_data, weight_list, possible_champion)
    # return jsonify(result)
    return "asdf"

def get_winning_rate_by_champion(user_lane,user_most):
    all_lane_by_champion = all_lane[user_lane-1]

    wining_rate_and_champion = {}

    for dictionary in all_lane_by_champion: 
        for most in user_most:
            if str(most["id"]) == dictionary["id"] and most["games"] >= 10:
                wining_rate_and_champion[dictionary["id"]] = most["winrate"]
                break
            else:
                wining_rate_and_champion[dictionary["id"]] = 25 # 최저 승률 보정

    return wining_rate_and_champion

def get_softmax_winning_rate_list(user_lane, user_most):

    winning_rate_and_champion = {}

    if user_lane  == 1: # top
        winning_rate_and_champion = get_winning_rate_by_champion(user_lane,user_most)
    elif user_lane  == 2: # jg
        winning_rate_and_champion = get_winning_rate_by_champion(user_lane,user_most)
    elif user_lane  == 3: # mid
        winning_rate_and_champion = get_winning_rate_by_champion(user_lane,user_most)
    elif user_lane  == 4: # ad
        winning_rate_and_champion = get_winning_rate_by_champion(user_lane,user_most)
    elif user_lane  == 5: # sup
        winning_rate_and_champion = get_winning_rate_by_champion(user_lane,user_most)

    number_of_champion = len(winning_rate_and_champion)

    winning_rate_list = list(winning_rate_and_champion.values())
    winning_rate_list = np.array(winning_rate_list, dtype=np.float64)

    print("dd", winning_rate_list)
    print(np.sum(winning_rate_list))
    max_value = np.max(winning_rate_list)
    winning_rate_list /= max_value # 정규화


    softmax_winning_rate_list = softmax(winning_rate_list)

    print("winning_rate_list", winning_rate_list, np.sum(winning_rate_list))
    print("softmax_winning_rate_list", softmax_winning_rate_list, np.sum(softmax_winning_rate_list))
    return softmax_winning_rate_list

# def weight_caculate(user_most):

#     most_champ = []
#     weight_list = []

#     # 판수가 10판이 넘고 승률이 50퍼가 넘는 챔피언만 추출
#     for most in user_most:
#         if most["games"] >= 10:
#             most_champ.append(most)

#     # 승률 X 판수
#     for most in most_champ:
#         weight = most["games"] * most["winrate"]
#         weight_list.append([most["id"], weight])




#     # weight 정규화
#     weight_column = [row[1] for row in weight_list]

#     # print("weight_column", weight_column)

#     max_value = max(weight_column)

    
#     for i in range(len(weight_column)):
#         weight_column[i] /= max_value 

#     # print("weight_column", weight_column)

#     softmax_value = softmax(weight_column)

#     # print("softmax_value", softmax_value)

#     for row in weight_list:
#         row[1] = softmax_value[weight_list.index(row)]

#     # print("weight_list", weight_list)   
#     return weight_list


def preprocess_input_data(client_json):
    input_data = np.zeros(10)
    input_data[:5] = client_json["theirTeam"]
    values = list(client_json["myTeam"].values())
    input_data[5:] = values

    return input_data

def model_predict(input_data, user_lane):
    input_data = np.expand_dims(input_data, axis=0)

    pred = np.zeros(0)
    print("user_lane", user_lane)

    if user_lane  == 1: # top
        pred = top_model.predict(input_data)
        temp = 3
    elif user_lane  == 2: # jg
        pred = jg_model.predict(input_data)
        temp = 2
    elif user_lane  == 3: # mid
        pred = mid_model.predict(input_data)
        temp = 2
    elif user_lane  == 4: # ad
        pred = ad_model.predict(input_data)
        temp = 3
    elif user_lane  == 5: # sup
        pred = sup_model.predict(input_data)
        temp = 2

    # 정규화
    pred /= temp
    min_value = -np.min(pred)
    pred += min_value
    pred = softmax(pred)

    print("pred", pred)

    return pred

def preprocess_pred(pred, ban_state, user_lane, user_most, input_data, weight_list, possible_champion):
    
    length_of_pred = len(pred[0])
    print("length_of_pred", length_of_pred)

    alpha = sum(tf.math.top_k(pred[0], k=3).values)


    pred_list = pred[0]
    softmax_winning_rate_list = get_softmax_winning_rate_list(user_lane, user_most)

    print("가중치 적용 전", pred_list)

    sorted_indices = tf.math.top_k(pred_list, k=length_of_pred).indices
    sorted_values = tf.math.top_k(pred_list, k=length_of_pred).values

    # pred_list = alpha * pred[0] + (1-alpha) * softmax_winning_rate_list

    # sorted_indices = tf.math.top_k(pred_list, k=length_of_pred).indices
    # sorted_values = tf.math.top_k(pred_list, k=length_of_pred).values
    # print("가중치 적용 후", pred_list)



    # # 가중치 사용
    # pred_value_and_id = {}
    # for i in range(length_of_pred):
    #     pred_value_and_id[all_lane[user_lane-1][sorted_indices[i]]["id"]] = float(sorted_values[i])
        


    # sorted_pred_value_and_id = sorted(pred_value_and_id.items(), key=lambda x: x[1], reverse=True)

    # sorted_champion = {}

    # # possible champion만 사용하기
    # for id in sorted_pred_value_and_id:
    #     if int(id[0]) in possible_champion:
    #         sorted_champion[id[0]] = id[1]

    # for i in range(length_of_pred):

    #     # ban 반영해서 챔피언 제거
    #     for ban in ban_state:
    #         if str(ban) in sorted_champion.keys():
    #             sorted_champion.pop(str(ban))

    #     # 픽되어있는 챔피언 추천에서 제거
    #     for picked_champion in input_data:
    #         if str(int(picked_champion)) in sorted_champion.keys():
    #             sorted_champion.pop(str(int(picked_champion)))

    
    top_3_champion = []
    bottom_2_champion = []


    # 가중치 없는 버전
    sorted_champion = {}  
    print("sorted_indices", sorted_indices)
    for i in range(length_of_pred):
        print(all_lane[user_lane-1][sorted_indices[i]])
        sorted_champion[all_lane[user_lane-1][sorted_indices[i]]["id"]] = all_lane[user_lane-1][sorted_indices[i]]["name"]

        # ban 반영해서 챔피언 제거
        for ban in ban_state:
            if str(ban) in sorted_champion.keys():
                sorted_champion.pop(str(ban))

        # 픽되어있는 챔피언 추천에서 제거
        for picked_champion in input_data:
            if str(int(picked_champion)) in sorted_champion.keys():
                sorted_champion.pop(str(int(picked_champion)))



    ## 출력부분
    top_3_champion = list(sorted_champion)[:3]
    bottom_2_champion = list(sorted_champion)[-2:]
    print("sorted_champion", sorted_champion)


    print("sorted_values", sorted_values)
    variance = np.var(sorted_values)
    standard_deviation = np.std(sorted_values)
    mean = np.mean(sorted_values)
    total_sum = np.sum(sorted_values)
    print("sum", total_sum)
    print("mean", mean)
    print("variance", variance)
    print("standard_deviation", standard_deviation)
    print("top3", top_3_champion)
    print("bottom2", bottom_2_champion)

    return top_3_champion + bottom_2_champion
        

def softmax(x):
    e_x = np.exp(x - np.max(x))  # 입력에서 최댓값을 빼줌으로써 수치 안정성 보장
    return e_x / np.sum(e_x)

if __name__ == '__main__':
    from waitress import serve

    serve(
        app,
        host="0.0.0.0",
        port=8082,
    )