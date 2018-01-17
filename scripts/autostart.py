import os
import time

class TorcsInstance:

	def start(self):
		print("Relaunch Torcs 1.0")
		self.__close()
		self.__sleep()
		os.system(u'torcs -nofuel -nodamage -nolaptime -vision &')
		self.__sleep()
		os.system(u'sh scripts/autostart.sh')

	def changeTrack(self):
		print("Changing Track 1.0")
		self.__close()
		self.__sleep()
		os.system(u'torcs -nofuel -nodamage -nolaptime -vision &')
		self.__sleep()
		os.system(u'sh scripts/random_autostart.sh')
		self.__close()
		self.__sleep()

	def __close(self):
		os.system(u'pkill torcs')

	def __sleep(self, seconds=2.0):
		time.sleep(seconds)
