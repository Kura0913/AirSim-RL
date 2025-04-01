from datetime import datetime
from Platform import Platform

def main():
    platform = Platform()
    drone_name = platform.load_drone_name()
    yaml_config = platform.load_yaml_config('config.yaml')
    testing_setting = yaml_config['testing_setting']
    folder_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    platform.test(drone_name, testing_setting, folder_name)

if __name__ == "__main__":
    main()