package edu.illinois.incr_ga;

import java.util.Random;

import org.apache.hadoop.io.RawComparator;

public class LongArrayWritableComparator implements
		RawComparator<LongArrayWritable> {
	private static Random r;

	public LongArrayWritableComparator() {
		r = new Random(System.nanoTime());
	}

	public int compare(byte[] arg0, int arg1, int arg2, byte[] arg3, int arg4,
			int arg5) {
		// Compare two longs randomly so that the output is shuffled randomly and
		// not according to their values
		if (r.nextBoolean())
			return -1;
		else
			return 1;
	}

	public int compare(LongArrayWritable arg0, LongArrayWritable arg1) {
		// Compare two longs randomly so that the output is shuffled randomly and
		// not according to their values
		if (r.nextBoolean())
			return -1;
		else
			return 1;
	}

}
