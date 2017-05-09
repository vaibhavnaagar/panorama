import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2, sys
from PIL import Image


def dlt(f, t, num_points=4):
	"""	Returns Homography matrix in which 'f' points are mapped to 't' points
		using Direct Linear Transform algorithm.
		t = Hf
	"""
	assert f.shape == t.shape
	num_points = f.shape[0]
	A = np.zeros((2*num_points, 9))
	for p in range(num_points):
		fh = np.array([f[p,0], f[p,1], 1])										# Homogenous coordinate of point p
		A[2*p] = np.concatenate(([0, 0, 0], -fh, t[p,1]*fh))					# [0' -wX' yX']
		A[2*p + 1] = np.concatenate((fh, [0, 0, 0], -t[p,0]*fh))				# [wX' 0' -xX']
	U, D, V = np.linalg.svd(A)
	H = V[8].reshape(3, 3)
	return H / H[-1,-1]

def project_image(H, img):
	""" Project each image coordinate (pixel) using Homography matrix 'H' and
		returns the image array of projected image
	"""
	p_img = np.zeros(img.shape)
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			p_xy = np.dot(H, [x, y, 1])
			p_xy = np.round(p_xy/p_xy[2]).astype(int)
			if (0 <= p_xy[0] < img.shape[0]) and (0 <= p_xy[1] < img.shape[1]):
				p_img[tuple(p_xy[:-1])] = img[x,y]
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			if np.all(p_img[x,y] == [0,0,0]) or np.all(p_img[x,y] == [255,255,255]):
				p_img[x,y] = img[x,y]
	return p_img


if __name__ == '__main__':
	#fp = np.array([[515,340], [515,554], [635,560], [638,340]])
	#tp = np.array([[514,340], [514,552], [626,552], [626,340]])

	#fp = np.array([[340,515], [554,515], [560,635], [340,638]])		# rev(x,y)
	#tp = np.array([[340,514], [552,514], [552,626], [340,626]])		# rev(x,y)

	#fp = np.array([[142,119], [128,129], [148,184], [180,181]])		# rev (x,y)
	#tp = np.array([[116,80], [89,75], [82,156], [124,177]])			# rev (x,y)
	#fp = np.array([[119,142], [129,128], [184,148], [181,180]])
	#tp = np.array([[80,116], [75,89], [156,82], [177,124]])

	fp = np.array(eval(sys.argv[2]))
	tp = np.array(eval(sys.argv[3]))
	projective_matrix = dlt(fp, tp)
	print("Homography matrix:\n", projective_matrix)

	### Check Homography Matrix ###
	v = np.dot(projective_matrix, [129,128, 1])
	print(v/v[2])


	## USING PIL ##				(Reverse the coordinates)
	"""
	img = Image.open(sys.argv[1])
	img.show()
	p_img = img.transform(img.size, Image.PERSPECTIVE, tuple(projective_matrix.ravel().tolist()[:-1]), Image.BICUBIC)
	p_img.show()
	p_img.save('transform_pil_1.jpeg')
	#"""

	## USING CV2 ##		(No need to reverse the coordinates)
	#"""
	img = mpimg.imread(sys.argv[1])
	cols,rows = img.shape[0], img.shape[1]
	p_img = cv2.warpPerspective(img, projective_matrix, (rows,cols))
	plt.imshow(p_img, interpolation="nearest")
	plt.show()
	if p_img.shape[2] == 4:
		p_img = cv2.cvtColor(p_img, cv2.COLOR_BGRA2RGB)*256
	else:
		p_img = cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB)
	cv2.imwrite(sys.argv[4], p_img)
	#"""

	## USING project_image() ##		(Reverse the coordinates)
	"""
	img = mpimg.imread(sys.argv[1])
	p_img = project_image(projective_matrix, img)
	plt.imshow(p_img, interpolation="nearest")
	plt.show()
	mpimg.imsave("transorm_1.jpeg", p_img)
	"""

	## Trash ##
	#img = cv2.imread('to.png')
	#b,g,r = cv2.split(img)           # get b,g,r
	#rgb_img = cv2.merge([r,g,b])     # switch it to rgb
