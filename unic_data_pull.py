import socket
import time
import numpy as np
import os

def check_num_files():
    savepath = os.path.dirname(__file__)
    num_files = sum([1 if 'raw_data' in f else 0 for f in os.listdir(savepath)])
    return num_files

def data_retrieval(duration:int):
    num_files = check_num_files()

    while True:
        print("Unicorn Recorder UDP Receiver Example")
        print("----------------------------")
        print()

        try:
            # Define the destination port
            port = 1001
            ip = '0.0.0.0'  # Listen on all interfaces
            print(f"Listening on port '{port}'. Retrieving data...")

            # Initialize UDP socket
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
                udp_socket.bind((ip, port))

                count = 0
                duration = dur  # duration of one data set in seconds (kinda)
                elec = 8
                sf = 250
                data_array_raw = np.empty((sf*duration, elec))

                # Acquisition loop
                while True:
                    data, addr = udp_socket.recvfrom(1024)  # Buffer size is 1024 bytes
                    if data:
                        # Display received data
                        data_array_raw[count,:] = np.array(data.decode('ascii').split(',')[:-1]).astype(float)
                        count += 1

                    # Check if specified duration has passed
                    if count == sf*duration: # sf*duration = 2000 data points
                        print(f"{duration} seconds elapsed. Total data received: {count}")
                        break
                    
        except socket.error as sock_ex:
            print(f"Socket Error: {sock_ex}")
        except ValueError:
            print("Please enter a valid port number.")
        except Exception as ex:
            print(f"Error: {ex}")
        finally:
            print("Data retrieval complete!")

        np.save(os.path.join(savepath, f'raw_data{num_files+1}'), data_array_raw.T)
        
        return data_array_raw.T

if __name__ == "__main__":
    data_retrieval()
