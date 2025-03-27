from Platform import Platform
from datetime import datetime


def main():
    platform = Platform()
    drone_name = platform.load_drone_name()
    yaml_config = platform.load_yaml_config('config.yaml')
    training_setting = yaml_config['training_setting']
    folder_name = datetime.now().strftime('%Y%m%d_%H%M%S')

    platform.train(drone_name, training_setting, folder_name)

if __name__ == "__main__":
    main()