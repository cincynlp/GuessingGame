import threading
import queue
import time

class Logger:
    def __init__(self, filename, separater="\n", queue_timeout=1):
        self.filename = filename
        self.queue = queue.Queue()
        self.timeout = queue_timeout
        self.writer_thread = threading.Thread(target=self._write_to_file)
        self.writer_thread.daemon = True  # Daemonize the thread to automatically exit when the main program exits
        self.writer_thread.start()
        self.separater = separater
        
    def _write_to_file(self):
        """Background thread function that writes data from the queue to the file."""
        with open(self.filename, 'a') as f:
            while True:
                try:
                    # Wait for data in the queue, timeout after a small period
                    output = self.queue.get(timeout=self.timeout)
                    if output == "DONE":  # Special signal to stop the thread
                        break
                    f.write(output + self.separater)
                    f.flush()  # Ensure the data is written immediately
                    time.sleep(0)
                except TimeoutError:
                    # Timeout reached, continue the loop. No output to write.
                    if self.queue.empty(): continue
                    else:
                        print("Timeout while fetching")
                        break
                except queue.Empty:
                    continue

    def log(self, message, print_to_bash = True):
        """Public method to log messages to the queue."""
        if print_to_bash:
            print(message)
        self.queue.put(message)

    def stop(self):
        """Stop the writer thread gracefully by sending the 'DONE' signal."""
        self.queue.put("DONE")
        self.writer_thread.join()  # Wait for the writer thread to finish