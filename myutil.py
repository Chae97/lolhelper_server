def preprocess_input_data(client_json):
    input_data = np.zeros(10)
    input_data[:5] = client_json["theirTeam"]
    values = list(client_json["myTeam"].values())
    input_data[5:] = values

    return input_data