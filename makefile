python=python3

files = results-2032/computer-hardware/computer-hardware-0.33-0.csv results-2032/computer-hardware/computer-hardware-0.33-friedman.csv results-2032/computer-hardware/computer-hardware-0.33-1.csv results-2032/computer-hardware/computer-hardware-0.33-friedman.csv results-2032/computer-hardware/computer-hardware-0.33-2.csv results-2032/computer-hardware/computer-hardware-0.33-friedman.csv results-2032/computer-hardware/computer-hardware-0.33-3.csv results-2032/computer-hardware/computer-hardware-0.33-friedman.csv results-2032/computer-hardware/computer-hardware-0.33-4.csv results-2032/computer-hardware/computer-hardware-0.33-friedman.csv

all	: $(files)
	@echo ALL done

delete	:
	rm -f $(files)
	@echo DELETE done

build	: delete all
	@echo BUILD done

results-2032/computer-hardware/computer-hardware-0.33-friedman.csv	: friedman.py results-2032/computer-hardware/computer-hardware-0.33-0.csv results-2032/computer-hardware/computer-hardware-0.33-1.csv results-2032/computer-hardware/computer-hardware-0.33-2.csv results-2032/computer-hardware/computer-hardware-0.33-3.csv results-2032/computer-hardware/computer-hardware-0.33-4.csv results-2032/computer-hardware/computer-hardware-0.33-5.csv results-2032/computer-hardware/computer-hardware-0.33-6.csv results-2032/computer-hardware/computer-hardware-0.33-7.csv results-2032/computer-hardware/computer-hardware-0.33-8.csv results-2032/computer-hardware/computer-hardware-0.33-9.csv results-2032/computer-hardware/computer-hardware-0.33-10.csv results-2032/computer-hardware/computer-hardware-0.33-11.csv results-2032/computer-hardware/computer-hardware-0.33-12.csv results-2032/computer-hardware/computer-hardware-0.33-13.csv results-2032/computer-hardware/computer-hardware-0.33-14.csv results-2032/computer-hardware/computer-hardware-0.33-15.csv results-2032/computer-hardware/computer-hardware-0.33-16.csv results-2032/computer-hardware/computer-hardware-0.33-17.csv results-2032/computer-hardware/computer-hardware-0.33-18.csv results-2032/computer-hardware/computer-hardware-0.33-19.csv
	$(python) friedman.py computer-hardware 0.33 2032

results-2032/computer-hardware/computer-hardware-0.33-0.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-0.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-0.csv computer-hardware 0.33-0 2032

results-2032/computer-hardware/computer-hardware-0.33-1.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-1.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-1.csv computer-hardware 0.33-1 2032

results-2032/computer-hardware/computer-hardware-0.33-2.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-2.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-2.csv computer-hardware 0.33-2 2032

results-2032/computer-hardware/computer-hardware-0.33-3.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-3.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-3.csv computer-hardware 0.33-3 2032

results-2032/computer-hardware/computer-hardware-0.33-4.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-4.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-4.csv computer-hardware 0.33-4 2032

results-2032/computer-hardware/computer-hardware-0.33-5.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-5.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-5.csv computer-hardware 0.33-5 2032

results-2032/computer-hardware/computer-hardware-0.33-6.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-6.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-6.csv computer-hardware 0.33-6 2032

results-2032/computer-hardware/computer-hardware-0.33-7.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-7.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-7.csv computer-hardware 0.33-7 2032

results-2032/computer-hardware/computer-hardware-0.33-8.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-8.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-8.csv computer-hardware 0.33-8 2032

results-2032/computer-hardware/computer-hardware-0.33-9.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-9.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-9.csv computer-hardware 0.33-9 2032

results-2032/computer-hardware/computer-hardware-0.33-10.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-10.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-10.csv computer-hardware 0.33-10 2032

results-2032/computer-hardware/computer-hardware-0.33-11.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-11.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-11.csv computer-hardware 0.33-11 2032

results-2032/computer-hardware/computer-hardware-0.33-12.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-12.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-12.csv computer-hardware 0.33-12 2032

results-2032/computer-hardware/computer-hardware-0.33-13.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-13.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-13.csv computer-hardware 0.33-13 2032

results-2032/computer-hardware/computer-hardware-0.33-14.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-14.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-14.csv computer-hardware 0.33-14 2032

results-2032/computer-hardware/computer-hardware-0.33-15.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-15.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-15.csv computer-hardware 0.33-15 2032

results-2032/computer-hardware/computer-hardware-0.33-16.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-16.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-16.csv computer-hardware 0.33-16 2032

results-2032/computer-hardware/computer-hardware-0.33-17.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-17.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-17.csv computer-hardware 0.33-17 2032

results-2032/computer-hardware/computer-hardware-0.33-18.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-18.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-18.csv computer-hardware 0.33-18 2032

results-2032/computer-hardware/computer-hardware-0.33-19.csv	: classification.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-19.csv
	$(python) regression.py datasets-2032/computer-hardware/computer-hardware-train-0.33.csv datasets-2032/computer-hardware/computer-hardware-test-0.33-19.csv computer-hardware 0.33-19 2032

