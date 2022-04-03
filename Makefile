vXXX:
	make -C vXXX

v500:
	make -C v500

v400:
	make -C v400

v704:
	make -C v704

us2:
	make -C us1

v601:
	make -C v601

v602:
	make -C v602

clean:
	make -C vXXX clean
	make -C v500 clean
	make -C v400 clean
	make -C v704 clean
	make -C us1 clean
	make -C v601 clean
	make -C v602 clean

.PHONY: clean vXXX clean v500 clean v400 clean v704 clean us1 clean v601 clean v602