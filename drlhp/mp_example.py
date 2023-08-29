import multiprocessing


def worker_function(number):
    print(f"Worker {number} is working.")


if __name__ == "__main__":
    # Create two processes
    process1 = multiprocessing.Process(target=worker_function, args=(1,))
    process2 = multiprocessing.Process(target=worker_function, args=(2,))

    # Start the processes
    process1.start()
    process2.start()

    # Wait for both processes to complete
    process1.join()
    process2.join()

    print("Both workers are done!")
