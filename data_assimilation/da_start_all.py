from data_assimilation import start_all
f = open("done.txt", "w+")
start_all('commoners population', 1, 0.001, f)
start_all('elite population', 2, 0.001, f)
start_all('nature', 3, 0.001, f)
start_all('wealth', 4, 0.001, f)

start_all('commoners population', 1, 0.002, f)
start_all('elite population', 2, 0.002, f)
start_all('nature', 3, 0.002, f)
start_all('wealth', 4, 0.002, f)

start_all('commoners population', 1, 0.01, f)
start_all('elite population', 2, 0.01, f)
start_all('nature', 3, 0.01, f)
start_all('wealth', 4, 0.01, f)

start_all('commoners population', 1, 0.02, f)
start_all('elite population', 2, 0.02, f)
start_all('nature', 3, 0.02, f)
start_all('wealth', 4, 0.02, f)

start_all('commoners population', 1, 0.1, f)
start_all('elite population', 2, 0.1, f)
start_all('nature', 3, 0.1, f)
start_all('wealth', 4, 0.1, f)
f.close()
