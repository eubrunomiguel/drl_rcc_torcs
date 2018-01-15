import os
import time

class TorcsInstance:

	def start(self):
		print("Relaunch Torcs 2.1")
		self.stop()
		self.__wait(2)
		self.__start()
		self.__scriptboot()

	def stop(self):
		os.system('pkill torcs')

	def __wait(self, seconds):
		time.sleep(seconds)

	def __scriptboot(self):
		self.wait(1.5)
		os.system(u'sh autostart.sh')
		self.wait(1.5)

	def __start(self):
		os.system(u'torcs -nofuel -nodamage -nolaptime &')

