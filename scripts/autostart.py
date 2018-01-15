import os
import time

class TorcsInstance:

	def start(self, random_track=False):
		print("Relaunch Torcs 2.0")
		os.system(u'pkill torcs')
		time.sleep(1.0)
		os.system(u'torcs -nofuel -nodamage -nolaptime -vision &')
		time.sleep(1.0)
		if random_track:
			print("opening random")
			os.system(u'sh random_autostart.sh')
		else:
			os.system(u'sh autostart.sh')