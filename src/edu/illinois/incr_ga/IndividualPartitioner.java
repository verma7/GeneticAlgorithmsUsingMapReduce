package edu.illinois.incr_ga;

import java.util.Random;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;

public class IndividualPartitioner implements
		Partitioner<LongArrayWritable, LongWritable> {
	// Partitions randomly independent of the passed <K, V>

	public int getPartition(LongArrayWritable arg0, LongWritable arg1,
			int numReducers) {
		int red = (int) ((Math.floor(Math.random() * 10000)) % numReducers);
//		System.out.println("Using partitioner: returning " + red);
		return red;
	}

	public void configure(JobConf arg0) {
		// TODO Auto-generated method stub

	}
}
