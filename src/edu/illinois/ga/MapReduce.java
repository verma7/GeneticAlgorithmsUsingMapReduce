package edu.illinois.ga;

import java.io.IOException;
import java.util.Iterator;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Partitioner;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.lib.IdentityReducer;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;


@SuppressWarnings("deprecation")
public class MapReduce extends Configured implements Tool {

	public static final int LONG_BITS = 64;

	public static String rootDir = "/home/verma7/";

	public static class IndividualPartitioner<LongArrayWritable, LongWritable> implements Partitioner<LongArrayWritable, LongWritable> {
		// Partitions randomly independent of the passed <K, V>
		Random rng;

		public void configure(JobConf arg0) {
			rng = new Random(System.nanoTime());
		}

		public int getPartition(LongArrayWritable arg0, LongWritable arg1, int numReducers) {
			return (Math.abs(rng.nextInt()) % numReducers);
		}		
	}	

	public static class InitialGAMapper extends MapReduceBase
	implements Mapper<LongArrayWritable,LongWritable, LongArrayWritable, LongWritable> {
		Random rng;		
		int LONGS_PER_ARRAY;
		LongWritable[] individual ;

		@Override
		public void configure(JobConf jc) {			
			LONGS_PER_ARRAY = Integer.parseInt(jc.get("ga.longsPerArray"));
			rng = new Random(System.nanoTime());
			individual = new LongWritable[LONGS_PER_ARRAY];
		}

		public void map(LongArrayWritable key, LongWritable value, OutputCollector<LongArrayWritable, LongWritable> oc, Reporter rep) throws IOException {
			
			for(int i=0; i<value.get(); i++) {
				// Generate initial individual
				for(int l=0; l<LONGS_PER_ARRAY; l++) {
					long ind = 0;
					for(int m=0; m < LONG_BITS; m++) {
						ind = ind | (rng.nextBoolean()? 0: 1);
						// Don't shift for the last bit
						if(m != LONG_BITS - 1)
							ind = ind << 1;						
					}
					individual[l] = new LongWritable(ind);
//					System.out.print(individual[l].get());
				}
				oc.collect(new LongArrayWritable(individual), new LongWritable(0));
			}
		}
	}

	public static class GAMapper extends MapReduceBase
	implements Mapper<LongArrayWritable,LongWritable, LongArrayWritable, LongWritable> {
		long max = -1;
		LongArrayWritable maxInd;
		private String mapTaskId = "";
		long fit = 0;
		JobConf conf;
		int pop = 1;
		@Override
		public void configure(JobConf job) {
			conf = job;
			mapTaskId = job.get("mapred.task.id");
			pop = Integer.parseInt(job.get("ga.populationPerMapper"));
		}

		long fitness(LongWritable[] individual) {
			long f=0;
			for(int i=0; i<individual.length; i++) {
				long mask = 1;
				for(int j=0; j<LONG_BITS; j++) {					
					f += ((individual[i].get() & mask) > 0)? 1 : 0;
					mask = mask << 1;
				}
			}
			//			System.err.println("Fitness of " + individual + " is " + f);
			return f;
		}

		int processedInd = 0;
		public void map(LongArrayWritable key, LongWritable value, OutputCollector<LongArrayWritable, LongWritable> oc, Reporter rep) throws IOException {
			// Compute the fitness for every individual
			LongWritable[] individual = key.getArray();
			fit = fitness(individual);
			// System.err.println(value + " : " + individual + " : " + fit);

			//Keep track of the maximum fitness
			if(fit > max) {
				max = fit;
				maxInd = new LongArrayWritable(individual);
			}
			oc.collect(key, new LongWritable(fit));
			processedInd++;
			if(processedInd == pop -1) {
				closeAndWrite();
			}
		}

		public void closeAndWrite() throws IOException {
			// At the end of Map(), write the best found individual to a file
			Path tmpDir = new Path( rootDir + "GA");
			Path outDir = new Path(tmpDir, "global-map");

			// HDFS does not allow multiple mappers to write to the same file, hence create one for each mapper
			Path outFile = new Path(outDir, mapTaskId);
			FileSystem fileSys = FileSystem.get(conf);
			SequenceFile.Writer writer = SequenceFile.createWriter(fileSys, conf, 
					outFile, LongArrayWritable.class, LongWritable.class, 
					CompressionType.NONE);

			// System.err.println("Max ind = " + maxInd.toString() + " : " + max);
			writer.append(maxInd, new LongWritable(max));
			writer.close();
		}

	}

	public static class GAReducer extends MapReduceBase
	implements Reducer<LongArrayWritable, LongWritable, LongArrayWritable, LongWritable> {

		int tournamentSize = 5;
		int LONGS_PER_ARRAY;
		LongWritable[][] tournamentInd;
		long[] tournamentFitness = new long[2*tournamentSize];

		int processedIndividuals=0;
		int r=0;
		LongArrayWritable[] ind = new LongArrayWritable[2];
		Random rng;
		int pop = 1;
		GAReducer() {
			rng = new Random(System.nanoTime());
		}
		@Override
		public void configure(JobConf jc) {			
			LONGS_PER_ARRAY = Integer.parseInt(jc.get("ga.longsPerArray"));
			tournamentInd = new LongWritable[2*tournamentSize][LONGS_PER_ARRAY];
			pop = Integer.parseInt(jc.get("ga.populationPerMapper"));
		}

		void crossover() {
			//Perform uniform crossover
			LongWritable[] ind1 = ind[0].getArray();
			LongWritable[] ind2 = ind[1].getArray();
			LongWritable[] newInd1 = new LongWritable[LONGS_PER_ARRAY];
			LongWritable[] newInd2 = new LongWritable[LONGS_PER_ARRAY];
			//			System.err.print("[GA] Crossing over " + ind[0] + " + " + ind[1]);

			for(int i=0; i<LONGS_PER_ARRAY; i++) {
				long i1 = 0, i2 = 0, mask = 1;
				for(int j=0; j<LONG_BITS; j++) {
					if(rng.nextDouble() > 0.5) {
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
			//			System.err.println("[GA] Got " + ind[0] + " + " + ind[1]);
		}

		LongWritable[] tournament(int startIndex) {
			// Tournament selection without replacement
			LongWritable[] tournamentWinner = null;
			long tournamentMaxFitness = -1;
			for(int j=0; j<tournamentSize; j++) {
				if(tournamentFitness[j] > tournamentMaxFitness) {
					tournamentMaxFitness = tournamentFitness[j];
					tournamentWinner = tournamentInd[j];
				}
			}
			return tournamentWinner;
		}

		OutputCollector<LongArrayWritable, LongWritable> _output;

		public void reduce(LongArrayWritable key, Iterator<LongWritable> values,
				OutputCollector<LongArrayWritable, LongWritable> output, Reporter rep)
		throws IOException {
			// Save the output collector for later use
			_output = output;

			while(values.hasNext()) {
				long fitness = values.next().get();
				tournamentInd[processedIndividuals%tournamentSize] = key.getArray();
				tournamentFitness[processedIndividuals%tournamentSize] = fitness;

				if ( processedIndividuals < tournamentSize ) {
					// Wait for individuals to join in the tournament and put them for the last round
					tournamentInd[processedIndividuals%tournamentSize + tournamentSize] = key.getArray();
					tournamentFitness[processedIndividuals%tournamentSize + tournamentSize] = fitness;
				} else {
					// Conduct a tournament over the past window
					ind[processedIndividuals%2] = new LongArrayWritable(tournament(processedIndividuals)); 

					if ((processedIndividuals - tournamentSize) %2 == 1) {					
						// Do crossover every odd iteration between successive individuals
						crossover();
						output.collect(ind[0], new LongWritable(0));
						output.collect(ind[1], new LongWritable(0));
					}
				}
				processedIndividuals++;
//				System.err.println(" " + processedIndividuals);
			}
			if(processedIndividuals == pop - 1) {
				closeAndWrite();
			}
		}

		public void closeAndWrite() {
			System.out.println("Closing reducer");
			// Cleanup for the last window of tournament
			for(int k=0; k<tournamentSize; k++) {
				// Conduct a tournament over the past window				
				ind[processedIndividuals%2] = new LongArrayWritable(tournament(processedIndividuals)); 

				if ((processedIndividuals - tournamentSize) %2 == 1) {					
					// Do crossover every odd iteration between successive individuals
					crossover();
					try {
						_output.collect(ind[0], new LongWritable(0));
						_output.collect(ind[1], new LongWritable(0));
					} catch (IOException e) {
						System.err.println("Exception in collector of reducer");
						e.printStackTrace();
					}
				}
				processedIndividuals++;
			}
		}
	}

	void launch(int numMaps, int numReducers, String jt, String dfs, int strLen, int pop, int iter) {
		int LONGS_PER_ARRAY = (int) Math.ceil(strLen / LONG_BITS);
		int it=0;
		while(true) {
			JobConf jobConf = new JobConf(getConf(), MapReduce.class);

			jobConf.setSpeculativeExecution(true);
			jobConf.setInputFormat(SequenceFileInputFormat.class);

			jobConf.setOutputKeyClass(LongArrayWritable.class);
			jobConf.setOutputValueClass(LongWritable.class);
			jobConf.setOutputFormat(SequenceFileOutputFormat.class);

			jobConf.set("ga.longsPerArray", LONGS_PER_ARRAY + "");

			jobConf.setNumMapTasks(numMaps);
			jobConf.setPartitionerClass(IndividualPartitioner.class);

			if (jt != null) { jobConf.set("mapred.job.tracker", jt); }
			if (dfs != null) { FileSystem.setDefaultUri(jobConf, dfs); }
			jobConf.setJobName("ga-mr-" + it);
			System.out.println("launching");

			Path tmpDir = new Path(rootDir + "GA");
			Path inDir = new Path(tmpDir, "iter" + it);
			Path outDir = new Path(tmpDir, "iter" + (it + 1));
			FileInputFormat.setInputPaths(jobConf, inDir);
			FileOutputFormat.setOutputPath(jobConf, outDir);

			FileSystem fileSys = null;
			try {
				fileSys = FileSystem.get(jobConf);
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			int populationPerMapper = pop/numMaps;
			jobConf.set("ga.populationPerMapper", populationPerMapper + "");

			if(it == 0) {
				// Initialization
				try {
					fileSys.delete(tmpDir, true);
				} catch(IOException ie) {
					System.out.println("Exception while deleting");
					ie.printStackTrace();
				}
				System.out.println("Deleting dir");

				for(int i=0; i < numMaps; ++i) {
					Path file = new Path(inDir, "part-"+String.format("%05d", i));
					SequenceFile.Writer writer = null;
					try {
						writer = SequenceFile.createWriter(fileSys, jobConf, 
								file, LongArrayWritable.class, LongWritable.class, CompressionType.NONE);
					}catch(Exception e) {
						System.out.println("Exception while instantiating writer");
						e.printStackTrace();
					}

					// Generate dummy input					
					LongWritable[] individual = new LongWritable[1];
					individual[0] = new LongWritable(populationPerMapper);
					try{
						writer.append(new LongArrayWritable(individual), new LongWritable(populationPerMapper));
					}catch(Exception e) {
						System.out.println("Exception while appending to writer");
						e.printStackTrace();
					}

					try{
						writer.close();
					} catch(Exception e) {
						System.out.println("Exception while closing writer");
						e.printStackTrace();
					}
					System.out.println("Writing dummy input for Map #" + i);
				}
				jobConf.setMapperClass(InitialGAMapper.class);
				jobConf.setReducerClass(IdentityReducer.class);
				jobConf.setNumReduceTasks(0);
			} // End of if it == 0
			else {
				jobConf.setMapperClass(GAMapper.class);
				jobConf.setReducerClass(GAReducer.class);
				jobConf.setNumReduceTasks(numReducers);
				try {
					fileSys.delete(outDir, true);
					fileSys.delete(new Path(tmpDir, "global-map"), true);
				} catch(IOException ie) {
					System.out.println("Exception while deleting");
					ie.printStackTrace();
				}
			}
			
			System.out.println("Starting Job");
			long startTime = System.currentTimeMillis();

			try {
				JobClient.runJob(jobConf);
			} catch (IOException e) {
				System.out.println("Exception while running job");
				e.printStackTrace();
			}


			LongWritable max = new LongWritable();
			LongArrayWritable maxInd = new LongArrayWritable();
			LongWritable finalMax = new LongWritable(-1);
			LongArrayWritable finalInd = null;

			// At the end of job, find out the best individual
			if(it > 0) {
				Path global = new Path(tmpDir, "global-map");

				FileStatus[] fs = null;
				SequenceFile.Reader reader = null;
				try {
					fs = fileSys.listStatus(global);
				} catch (IOException e) {
					System.out.println("Exception while instantiating reader in find winner");
					e.printStackTrace();
				}

				for(int i=0; i<fs.length; i++) {
					Path inFile = fs[i].getPath();
					try {
						reader = new SequenceFile.Reader(fileSys, inFile,
								jobConf);
					} catch (IOException e) {
						System.out.println("Exception while instantiating reader");
						e.printStackTrace();
					}

					try {
						while(reader.next(maxInd, max)) {
							if(max.get() > finalMax.get()) {
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

				/*			System.out.println("The best individual is : (" + finalInd + " , " + finalMax.get() + ")");
				System.out.println("Job Finished in "+
						(System.currentTimeMillis() - startTime)/1000.0 + " seconds");
				 */
				System.out.println("GA:" + it + ":" + LONGS_PER_ARRAY * LONG_BITS + ":" + pop + ":" + finalMax.get() + ":" + (System.currentTimeMillis() - startTime));
				if(finalMax.get() >= LONGS_PER_ARRAY * LONG_BITS - 10)
					break;
			}
			it++;
		}
	}

	/**
	 * Launches all the tasks in order.
	 */
	public int run(String[] args) throws Exception {
		if (args.length != 5) {
			System.err.println("Usage: GeneticMR <nMaps> <nReducers> <variables> <nIterations> <popTimesNlogN>");
			ToolRunner.printGenericCommandUsage(System.err);
			return -1;
		}

		int	nMaps = Integer.parseInt(args[0]);
		int	nReducers = Integer.parseInt(args[1]);
		int strLen = Integer.parseInt(args[2]);
		int iter = Integer.parseInt(args[3]);
		int pop = (int) Math.ceil(Integer.parseInt(args[4]) * strLen * Math.log(strLen) / Math.log(2));
		System.out.println("Number of Maps = " + nMaps);  
		launch(nMaps, nReducers, null, null, strLen, pop, iter);

		return 0;
	}

	public static void main(String[] argv) throws Exception {
		int res = ToolRunner.run(new Configuration(), new MapReduce(), argv);
		System.exit(res);
	}
}
