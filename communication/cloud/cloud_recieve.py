from ecci_sdk import Client
import threading
from threading import Thread
import time

if __name__ == '__main__':
    ecci_client = Client()
    mqtt_thread = threading.Thread(target=ecci_client.initialize)
    mqtt_thread.start()
    ecci_client.wait_for_ready()
    print('cloud start --------')
    while True :
        ##############################################################
        edge_data = ecci_client.get_sub_data_payload_queue().get()

        print("#############trasmission time is:", (time.time() - edge_data["time"])*1000)
        print("#############recieve feature map from edge",edge_data)
        ##############################################################