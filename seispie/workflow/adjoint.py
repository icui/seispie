from seispie.workflow.base import base
from time import time
import numpy as np

class adjoint(base):
	""" forward simulation
	"""

	def setup(self):
		""" initialize
		"""
		pass
	
	def run(self):
		""" start workflow
		"""
		solver = self.solver
		solver.import_model(1)
		solver.import_sources()
		solver.import_stations()
		solver.import_traces()

		start = time()
		solver.import_model(0)
		gradient, _, _ = solver.compute_gradient()
		
		if not self.mpi or self.mpi.rank() == 0:
			print('')
			print('Elapsed time: %.2fs' % (time() - start))
			solver.export_field(gradient, 'kmu')

	@property
	def modules(self):
		return ['solver', 'postprocess']