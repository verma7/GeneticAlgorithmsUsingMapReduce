// Simple genetic algorithm using the new API and the incremental mapreduce
package edu.illinois.incr_ga;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

@SuppressWarnings("deprecation")
public class IncMapReduce {
	private static final Log LOG = LogFactory
			.getLog(IncMapReduce.class.getName());
	public static final int LONG_BITS = 64;

	public static int LONGS_PER_ARRAY = 1000;

	public static int POPULATION = 6000;

	public static String rootDir = "/home/verma7/";

	public static class InitialIncGAMapper extends
			Mapper<LongArrayWritable, LongWritable, LongArrayWritable, LongWritable> {
		Random rng;

		LongWritable[] individual;

		@Override
		public void setup(Context ctx) {
			Configuration c = ctx.getConfiguration();
			rng = new Random(System.nanoTime());
			individual = new LongWritable[LONGS_PER_ARRAY];
		}

		public void map(LongArrayWritable key, LongWritable value, Context context)
		// OutputCollector<LongArrayWritable, LongWritable> oc, Reporter rep)
				throws IOException {

			for (int i = 0; i < value.get(); i++) {
				// Generate initial individual
				for (int l = 0; l < LONGS_PER_ARRAY; l++) {
					long ind = 0;
					for (int m = 0; m < LONG_BITS; m++) {
						ind = ind | (rng.nextBoolean() ? 0 : 1);
						// Don't shift for the last bit
						if (m != LONG_BITS - 1)
							ind = ind << 1;
					}
					individual[l] = new LongWritable(ind);
					// System.out.print(individual[l].get());
				}
				try {
					context.write(new LongArrayWritable(individual), new LongWritable(0));
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}

	public static class IncGAMapper extends
			Mapper<LongArrayWritable, LongWritable, LongArrayWritable, LongWritable> {
		long max = -1;

		LongArrayWritable maxInd;

		private String mapTaskId = "";

		long fit = 0;

		Configuration conf;

		int pop = POPULATION;

		@Override
		public void setup(Context context) {
			conf = context.getConfiguration();
			mapTaskId = conf.get("mapred.task.id");
			System.out.println("Partitioner: "
					+ context.getConfiguration().get("mapred.partitioner.class") + ", "
					+ context.getConfiguration().get("mapreduce.partitioner.class"));

			// pop = Integer.parseInt(conf.get("ga.populationPerMapper"));
		}

		long fitness(LongWritable[] individual) {
			long f = 0;
			for (int i = 0; i < individual.length; i++) {
				long mask = 1;
				for (int j = 0; j < LONG_BITS; j++) {
					f += ((individual[i].get() & mask) > 0) ? 1 : 0;
					mask = mask << 1;
				}
			}
			// System.err.println("Fitness of " + individual + " is " + f);
			return f;
		}

		int processedInd = 0;

		public void map(LongArrayWritable key, LongWritable value, Context context)
		// OutputCollector<LongArrayWritable, LongWritable> oc, Reporter rep)
				throws IOException {
			// Compute the fitness for every individual
			LongWritable[] individual = key.getArray();
			fit = fitness(individual);
			// System.err.println(value + " : " + individual + " : " + fit);

			// Keep track of the maximum fitness
			if (fit > max) {
				max = fit;
				maxInd = new LongArrayWritable(individual);
			}
			try {
				context.write(key, new LongWritable(fit));
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			processedInd++;
			if (processedInd == pop - 1) {
				closeAndWrite();
			}
		}

		public void closeAndWrite() throws IOException {
			// At the end of Map(), write the best found individual to a file
			Path tmpDir = new Path(rootDir + "GA");
			Path outDir = new Path(tmpDir, "global-map");

			// HDFS does not allow multiple mappers to write to the same file,
			// hence create one for each mapper
			Path outFile = new Path(outDir, mapTaskId);
			FileSystem fileSys = FileSystem.get(conf);
			SequenceFile.Writer writer = SequenceFile.createWriter(fileSys, conf,
					outFile, LongArrayWritable.class, LongWritable.class,
					CompressionType.NONE);

			// System.err.println("Max ind = " + maxInd.toString() + " : " +
			// max);
			writer.append(maxInd, new LongWritable(max));
			writer.close();
		}

	}

	public static class IncGAReducer extends
			Reducer<LongArrayWritable, LongWritable, LongArrayWritable, LongWritable> {

		int tournamentSize = 5;

		int LONGS_PER_ARRAY;

		LongWritable[][] tournamentInd;

		long[] tournamentFitness = new long[2 * tournamentSize];

		int processedIndividuals = 0;

		int r = 0;

		LongArrayWritable[] ind = new LongArrayWritable[2];

		Random rng;

		int pop = 1;

		IncGAReducer() {
			rng = new Random(System.nanoTime());
		}

		@Override
		public void setup(Context ctx) {
			tournamentInd = new LongWritable[2 * tournamentSize][LONGS_PER_ARRAY];
			pop = POPULATION;
		}

		void crossover() {
			// Perform uniform crossover
			LongWritable[] ind1 = ind[0].getArray();
			LongWritable[] ind2 = ind[1].getArray();
			LongWritable[] newInd1 = new LongWritable[LONGS_PER_ARRAY];
			LongWritable[] newInd2 = new LongWritable[LONGS_PER_ARRAY];
			// System.err.print("[GA] Crossing over " + ind[0] + " + " +
			// ind[1]);

			for (int i = 0; i < LONGS_PER_ARRAY; i++) {
				long i1 = 0, i2 = 0, mask = 1;
				for (int j = 0; j < LONG_BITS; j++) {
					if (rng.nextDouble() > 0.5) {
						i2 |= ind2[i].get() & mask;
						i1 |= ind1[i].get() & mask;
					} else {
						i1 |= ind2[i].get() & mask;
						i2 |= ind1[i].get() & mask;
					}
					mask = mask << 1;
				}
				newInd1[i] = new LongWritable(i1);
				newInd2[i] = new LongWritable(i2);
			}

			ind[0] = new LongArrayWritable(newInd1);
			ind[1] = new LongArrayWritable(newInd2);
			// System.err.println("[GA] Got " + ind[0] + " + " + ind[1]);
		}

		LongWritable[] tournament(int startIndex) {
			// Tournament selection without replacement
			LongWritable[] tournamentWinner = null;
			long tournamentMaxFitness = -1;
			for (int j = 0; j < tournamentSize; j++) {
				if (tournamentFitness[j] > tournamentMaxFitness) {
					tournamentMaxFitness = tournamentFitness[j];
					tournamentWinner = tournamentInd[j];
				}
			}
			return tournamentWinner;
		}

		HashMap<LongArrayWritable, LongWritable> hashMap = new HashMap<LongArrayWritable, LongWritable>();

		public void reduce(LongArrayWritable key, Iterator<LongWritable> values,
				Context context)
		// OutputCollector<LongArrayWritable, LongWritable> output, Reporter
				// rep)
				throws IOException {

			while (values.hasNext()) {
				long fitness = values.next().get();
				tournamentInd[processedIndividuals % tournamentSize] = key.getArray();
				tournamentFitness[processedIndividuals % tournamentSize] = fitness;

				if (processedIndividuals < tournamentSize) {
					// Wait for individuals to join in the tournament and put
					// them for the last round
					tournamentInd[processedIndividuals % tournamentSize + tournamentSize] = key
							.getArray();
					tournamentFitness[processedIndividuals % tournamentSize
							+ tournamentSize] = fitness;
				} else {
					// Conduct a tournament over the past window
					ind[processedIndividuals % 2] = new LongArrayWritable(
							tournament(processedIndividuals));

					if ((processedIndividuals - tournamentSize) % 2 == 1) {
						// Do crossover every odd iteration between successive
						// individuals
						crossover();
						hashMap.put(ind[0], new LongWritable(0));
						hashMap.put(ind[1], new LongWritable(0));
					}
				}
				processedIndividuals++;
				// System.err.println(" " + processedIndividuals);
			}
			if (processedIndividuals == pop - 1) {
				closeAndWrite();
			}
		}

		public void closeAndWrite() {
			System.out.println("Closing reducer");
			// Cleanup for the last window of tournament
			for (int k = 0; k < tournamentSize; k++) {
				// Conduct a tournament over the past window
				ind[processedIndividuals % 2] = new LongArrayWritable(
						tournament(processedIndividuals));

				if ((processedIndividuals - tournamentSize) % 2 == 1) {
					// Do crossover every odd iteration between successive
					// individuals
					crossover();
					hashMap.put(ind[0], new LongWritable(0));
					hashMap.put(ind[1], new LongWritable(0));
				}
				processedIndividuals++;
			}
		}

		// Override the run function for incremental functionality
		public void run(Context context) throws IOException, InterruptedException {
			while (context.nextKey()) {
				LongArrayWritable key = context.getCurrentKey();
				/*
				 * if (!hashMap.containsKey(key)) { hashMap.put(new
				 * LongArrayWritable(key), new LongWritable(0)); }
				 */
				// System.out.println("Calling user defined reduce funciton");
				reduce(key, context.getValues(), context);
			}

			LOG.info("INCRED: Reducer starting output to file");
			// sort keyset
			try {
				for (Object key : hashMap.keySet()) {
					// System.out.println("Writing output : " + key + " ");
					context.write((LongArrayWritable) (key), ((LongWritable) hashMap
							.get(key)));
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
			LOG.info("INCRED: Reducer done writing to output");
		}
	}

	static void launch(int numMaps, int numReducers, int iter, Configuration conf2) {
		int it = 0;
		while (true) {
			Job jobConf = null;
			Configuration conf = null;
			try {
				conf = new Configuration(conf2);
				conf2.setIncrementalReduction(true);
				jobConf = new Job(conf2);
			} catch (IOException e2) {
				// TODO Auto-generated catch block
				e2.printStackTrace();
			}

			// conf.setSpeculativeExecution(true);
			conf.setIncrementalReduction(true);
			conf2.setIncrementalReduction(true);
			jobConf.setJarByClass(IncMapReduce.class);
			jobConf.setInputFormatClass(SequenceFileInputFormat.class);
			jobConf.setMapOutputKeyClass(LongArrayWritable.class);
			jobConf.setMapOutputValueClass(LongWritable.class);
			jobConf.setOutputKeyClass(LongArrayWritable.class);
			jobConf.setOutputValueClass(LongWritable.class);
			jobConf.setOutputFormatClass(SequenceFileOutputFormat.class);

			jobConf.setGroupingComparatorClass(LongArrayWritableComparator.class);
			jobConf.setSortComparatorClass(LongArrayWritableComparator.class);
			jobConf.setPartitionerClass(IndividualPartitioner.class);

			conf.setClass("mapred.partitioner.class", IndividualPartitioner.class,
					Partitioner.class);
			// jobConf.setNumMapTasks(numMaps);

			jobConf.setJobName("iga-mr-" + it);
			System.out.println("launching");

			Path tmpDir = new Path(rootDir + "IGA");
			Path inDir = new Path(tmpDir, "iter" + it);
			Path outDir = new Path(tmpDir, "iter" + (it + 1));
			try {
				FileInputFormat.setInputPaths(jobConf, inDir);
			} catch (IOException e2) {
				// TODO Auto-generated catch block
				e2.printStackTrace();
			}
			FileOutputFormat.setOutputPath(jobConf, outDir);

			FileSystem fileSys = null;
			try {
				fileSys = FileSystem.get(conf);
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}

			if (it == 0) {
				// Initialization
				try {
					fileSys.delete(tmpDir, true);
				} catch (IOException ie) {
					System.out.println("Exception while deleting");
					ie.printStackTrace();
				}
				System.out.println("Deleting dir");

				for (int i = 0; i < numMaps; ++i) {
					Path file = new Path(inDir, "part-" + String.format("%05d", i));
					SequenceFile.Writer writer = null;
					try {
						writer = SequenceFile.createWriter(fileSys, conf, file,
								LongArrayWritable.class, LongWritable.class,
								CompressionType.NONE);
					} catch (Exception e) {
						System.out.println("Exception while instantiating writer");
						e.printStackTrace();
					}

					// Generate dummy input
					LongWritable[] individual = new LongWritable[1];
					individual[0] = new LongWritable(POPULATION);
					try {
						writer.append(new LongArrayWritable(individual), new LongWritable(
								POPULATION));
					} catch (Exception e) {
						System.out.println("Exception while appending to writer");
						e.printStackTrace();
					}

					try {
						writer.close();
					} catch (Exception e) {
						System.out.println("Exception while closing writer");
						e.printStackTrace();
					}
					System.out.println("Writing dummy input for Map #" + i);
				}
				jobConf.setMapperClass(InitialIncGAMapper.class);
				// jobConf.setReducerClass(IdentityReducer.class);
				jobConf.setNumReduceTasks(0);
			} // End of if it == 0
			else {
				jobConf.setMapperClass(IncGAMapper.class);
				jobConf.setReducerClass(IncGAReducer.class);
				jobConf.setNumReduceTasks(numReducers);
				try {
					fileSys.delete(outDir, true);
					fileSys.delete(new Path(tmpDir, "global-map"), true);
				} catch (IOException ie) {
					System.out.println("Exception while deleting");
					ie.printStackTrace();
				}
			}

			System.out.println("Starting Job");
			long startTime = System.currentTimeMillis();

			try {
				jobConf.waitForCompletion();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (ClassNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				System.out.println("Exception while running job");
				e.printStackTrace();
			}

			LongWritable max = new LongWritable();
			LongArrayWritable maxInd = new LongArrayWritable();
			LongWritable finalMax = new LongWritable(-1);
			LongArrayWritable finalInd = null;

			// At the end of job, find out the best individual
			if (it > 0) {
				Path global = new Path(tmpDir, "global-map");

				FileStatus[] fs = null;
				SequenceFile.Reader reader = null;
				try {
					fs = fileSys.listStatus(global);
				} catch (IOException e) {
					System.out
							.println("Exception while instantiating reader in find winner");
					e.printStackTrace();
				}

				for (int i = 0; i < fs.length; i++) {
					Path inFile = fs[i].getPath();
					try {
						reader = new SequenceFile.Reader(fileSys, inFile, conf);
					} catch (IOException e) {
						System.out.println("Exception while instantiating reader");
						e.printStackTrace();
					}

					try {
						while (reader.next(maxInd, max)) {
							if (max.get() > finalMax.get()) {
								finalMax = max;
								finalInd = maxInd;
							}
						}
					} catch (IOException e) {
						System.out.println("Exception while reading from reader");
						e.printStackTrace();
					}
					try {
						reader.close();
					} catch (IOException e) {
						System.out.println("Exception while closing reader");
						e.printStackTrace();
					}
				}

				/*
				 * System.out.println("The best individual is : (" + finalInd + " , " +
				 * finalMax.get() + ")"); System.out.println("Job Finished in "+
				 * (System.currentTimeMillis() - startTime)/1000.0 + " seconds");
				 */
				System.out.println("GA:" + it + ":" + LONGS_PER_ARRAY * LONG_BITS + ":"
						+ POPULATION + ":" + finalMax.get() + ":"
						+ (System.currentTimeMillis() - startTime));
				if (it == iter - 1)
					break;
			}
			it++;
		}
	}

	public static void main(String[] argv) throws Exception {
		Configuration conf = new Configuration();
		conf.setIncrementalReduction(true);
		String[] args = new GenericOptionsParser(conf, argv).getRemainingArgs();
		if (args.length != 3) {
			System.err.println("Usage: GeneticMR <nMaps> <nReducers> <nIterations>");
			System.exit(2);
		}

		int nMaps = Integer.parseInt(args[0]);
		int nReducers = Integer.parseInt(args[1]);
		int iter = Integer.parseInt(args[2]);
		System.out.println("Number of Maps = " + nMaps);
		launch(nMaps, nReducers, iter, conf);

		return;
	}
}
