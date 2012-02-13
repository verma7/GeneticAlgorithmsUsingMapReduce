package edu.illinois.incr_ga;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.WritableComparable;

public class LongArrayWritable implements WritableComparable<LongArrayWritable> {
	private LongWritable[] values;
	private static Random r;

	public LongArrayWritable() {
		r = new Random(System.nanoTime());
	}

	public LongArrayWritable(LongWritable[] iw) {
		r = new Random(System.nanoTime());
		values = iw.clone();
	}

	public LongArrayWritable(LongArrayWritable law) {
		values = law.values.clone();
	}

	public LongWritable[] getArray() {
		return values;
	}

	@Override
	public String toString() {
		String str = "Length: " + values.length + " : ";
		for (int i = 0; i < values.length; i++) {
			str += values[i].get() + "|";
		}
		return str;
	}

	public void readFields(DataInput in) {
		try {
			int len = in.readInt();
//			System.out.println("Deserializing: " + in.toString() + " of length "
					//+ len);
			values = new LongWritable[len]; // construct values
			for (int i = 0; i < values.length; i++) {
				LongWritable value = new LongWritable();
//				System.out.println("Trying to read longwritable " + i);
				value.readFields(in); // read a value
				values[i] = value; // store it in values
			}
		} catch (Exception ie) {
			values = new LongWritable[1];
			values[0] = new LongWritable(-1);
			System.err.println("Can't deserialize: " + ie.getMessage());
		}
	}

	public void write(DataOutput out) throws IOException {
//		System.out.println("Serializing: " + this);
		out.writeInt(values.length); // write values
		for (int i = 0; i < values.length; i++) {
			values[i].write(out);
		}
	}

	public int compareTo(LongArrayWritable arg0) {
		// Compare two longs randomly so that the output is shuffled randomly and
		// not according to their values
		if (r.nextBoolean())
			return -1;
		else
			return 1;
	}

	/*
	 * public int compareTo(LongArrayWritable o) {
	 * 
	 * }
	 */
}