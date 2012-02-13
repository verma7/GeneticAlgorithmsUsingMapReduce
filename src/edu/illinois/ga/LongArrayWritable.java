package edu.illinois.ga;

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

		public LongWritable[] getArray() {
			return values;
		}

		@Override
		public String toString() {
			String str = "";
			for(int i=0; i<values.length; i++) {
				str += values[i].get() + "|";
			}
			return str;
		}

		public void readFields(DataInput in) throws IOException {
			values = new LongWritable[in.readInt()];          // construct values
			for (int i = 0; i < values.length; i++) {
				LongWritable value = new LongWritable();
				value.readFields(in);                       // read a value
				values[i] = value;                          // store it in values
			}
		}

		public void write(DataOutput out) throws IOException {
			out.writeInt(values.length);                 // write values
			for (int i = 0; i < values.length; i++) {
				values[i].write(out);
			}
		}

		public int compareTo(LongArrayWritable o) {			
			// Compare two longs randomly so that the output is shuffled randomly and not according to their values
			if(r.nextBoolean())
				return -1;
			else
				return 1;
		}
	}