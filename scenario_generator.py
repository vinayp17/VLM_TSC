import json

# Define the configurations and parameters
configurations = {
    "BASELINE_2048": {
        "round_to": 2,
        "num_epochs": 2,
        "context_length": 2048,
        "data_repr": "BASELINE"
    },
    "BASELINE_4096": {
        "round_to": 2,
        "num_epochs": 2,
        "context_length": 4096,
        "data_repr": "BASELINE"
    },
    "WITH_STATS_2048": {
        "round_to": 2,
        "num_epochs": 2,
        "context_length": 2048,
        "data_repr": "WITH_STATS"
    },
    "WITH_STATS_4096": {
        "round_to": 2,
        "num_epochs": 2,
        "context_length": 4096,
        "data_repr": "WITH_STATS"
    }

}

# Function to generate JSON files for each configuration
def generate_json_configs(configurations):
    combined_config = {}
    for config_name, params in configurations.items():
        config_key = f"{config_name}"
        combined_config[config_key] = {
                "round_to": params["round_to"],
                "num_epochs": params["num_epochs"],
                "context_length": params["context_length"],
                "data_repr": params["data_repr"]
        }

    # Write the combined configuration to a single JSON file
    with open('combined_configurations.json', 'w') as json_file:
        json.dump(combined_config, json_file, indent=4)
    print("Generated combined_configurations.json")


# Generate the JSON configuration files
generate_json_configs(configurations)
