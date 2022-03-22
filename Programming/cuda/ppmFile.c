/****************************************************************
 *
 * ppm.c
 *
 * Read and write PPM files.  Only works for "raw" format.
 *
 * AF970205
 *
 ****************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "ppmFile.h"

/************************ private functions ****************************/



	

	/* die gracelessly */

	static void
	die(char *message)
	{
	  fprintf(stderr, "ppm: %s\n", message);
	  exit(1);
	}


	/* check a dimension (width or height) from the image file for reasonability */

	static void
	checkDimension(int dim)
	{
	  if (dim < 1 || dim > 4000) 
		die("file contained unreasonable width or height");
	}


	/* read a header: verify format and get width and height */

	static void
	readPPMHeader(FILE *fp, int *width, int *height)
	{
	  char ch;
	  int  maxval;

	  if (fscanf(fp, "P%c\n", &ch) != 1 || ch != '6') 
		die("file is not in ppm raw format; cannot read");

	  /* skip comments */
	  ch = getc(fp);
	  while (ch == '#')
		{
		  do {
		ch = getc(fp);
		  } while (ch != '\n');	/* read to the end of the line */
		  ch = getc(fp);            /* thanks, Elliot */
		}

	  if (!isdigit(ch)) die("cannot read header information from ppm file");

	  ungetc(ch, fp);		/* put that digit back */

	  /* read the width, height, and maximum value for a pixel */
	  fscanf(fp, "%d%d%d\n", width, height, &maxval);

	  if (maxval != 255) die("image is not true-color (24 bit); read failed");
	  
	  checkDimension(*width);
	  checkDimension(*height);
	}

	/************************ exported functions ****************************/

	Image *
	ImageCreate(int width, int height)
	{
	  Image *image = (Image *) malloc(sizeof(Image));

	  if (!image) die("cannot allocate memory for new image");

	  image->width  = width;
	  image->height = height;
	  image->data   = (unsigned char *) malloc(width * height * 3);

	  if (!image->data) die("cannot allocate memory for new image");

	  return image;
	}
	  

	Image *
	ImageRead(const char *filename)
	{
	  int width, height, num, size;

	  Image *image = (Image *) malloc(sizeof(Image));
	  FILE  *fp    = fopen(filename, "r");

	  if (!image) die("cannot allocate memory for new image");
	  if (!fp)    die("cannot open file for reading");

	  readPPMHeader(fp, &width, &height);

	  size          = width * height * 3;
	  image->data   = (unsigned  char*) malloc(size);
	  image->width  = width;
	  image->height = height;

	  if (!image->data) die("cannot allocate memory for new image");

	  num = fread((void *) image->data, 1, (size_t) size, fp);

	  if (num != size) die("cannot read image data from file");

	  fclose(fp);

	  return image;
	}


	void ImageWrite(Image *image, const char *filename)
	{
	  int num;
	  int size = image->width * image->height * 3;

	  FILE *fp = fopen(filename, "w");

	  if (!fp) die("cannot open file for writing");

	  fprintf(fp, "P6\n%d %d\n%d\n", image->width, image->height, 255);

	  num = fwrite((void *) image->data, 1, (size_t) size, fp);

	  if (num != size) die("cannot write image data to file");

	  fclose(fp);
	}  


	int
	ImageWidth(Image *image)
	{
	  return image->width;
	}


	int
	ImageHeight(Image *image)
	{
	  return image->height;
	}


	void   
	ImageClear(Image *image, unsigned char red, unsigned char green, unsigned char blue)
	{
	  int i;
	  int pix = image->width * image->height;

	  unsigned char *data = image->data;

	  for (i = 0; i < pix; i++)
		{
		  *data++ = red;
		  *data++ = green;
		  *data++ = blue;
		}
	}

	void
	ImageSetPixel(Image *image, int x, int y, int chan, unsigned char val)
	{
	  int offset = (y * image->width + x) * 3 + chan;

	  image->data[offset] = val;
	}


	unsigned  char
	ImageGetPixel(Image *image, int x, int y, int chan)
	{
	  int offset = (y * image->width + x) * 3 + chan;

	  return image->data[offset];
	}

