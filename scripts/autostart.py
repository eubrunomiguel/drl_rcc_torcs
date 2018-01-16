import os
import time

class TorcsInstance:

	def start(self, random_track=False):
		print("Relaunch Torcs 2.1")
		os.system(u'pkill torcs')
		time.sleep(2.0)
		os.system(u'torcs -nofuel -nodamage -nolaptime -vision &')
		time.sleep(2.0)
		if random_track is True:
			os.system(u'sh scripts/random_autostart.sh')
		else:
			os.system(u'sh scripts/autostart.sh')