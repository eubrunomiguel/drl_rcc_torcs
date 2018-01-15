import os
import time

class TorcsInstance:

	def start(self):
		print("Relaunch Torcs 2.0")
		os.system(u'pkill torcs')
		time.sleep(1.0)
		os.system(u'torcs -nofuel -nodamage -nolaptime -vision &')
		time.sleep(1.0)
		os.system(u'sh autostart.sh')