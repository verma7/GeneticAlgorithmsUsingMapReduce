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
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.lib.IdentityReducer;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
@SuppressWarnings("deprecation")

public class CGA extends Configured implements Tool {

	public static final int LONG_BITS = 64;
	public static final int LONGS_PER_ARRAY = 12;

	public static String rootDir = "/home/verma7/";

	public static class InitialCGAMapper extends MapReduceBase
	implements Mapper<LongWritable,LongArrayWritable, LongWritable, LongArrayWritable> {
		LongWritable[] individual;
		long numSplits = -1;
		int TOURNAMENT_SIZE;

		@Override
		public void configure(JobConf conf) {
			numSplits = Long.parseLong(conf.get("cga.numSplits"));
			TOURNAMENT_SIZE = Integer.parseInt(conf.get("cga.tournamentSize"));
			individual[0] = new LongWritable(TOURNAMENT_SIZE);
		}

		public InitialCGAMapper() {
			individual = new LongWritable[(LONGS_PER_ARRAY-1)*LONG_BITS + 1];
		 
		}

		public void map(LongWritable key, LongArrayWritable value, OutputCollector<LongWritable, LongArrayWritable> oc, Reporter rep) throws IOException {
			long mid = Long.MAX_VALUE / 2;
			for(int i=0; i<value.getArray()[0].get(); i++) {
				// Generate initial split
				for(int l=0; l<(LONGS_PER_ARRAY - 1) * LONG_BITS; l++) {
					individual[l+1] = new LongWritable(mid);
					//					System.err.print(mid + " ");
				}
				oc.collect(new LongWritable(key.get()*numSplits + i), new LongArrayWritable(individual));
			}
		}
	}

	// Gets as input <SplitNum, ProbabilityVectorSplit> and outputs <SplitNum, [ 1, Ind1 ]> and <SplitNum, [ 2, Ind2 ]>
	public static class CGAMapper extends MapReduceBase
	implements Mapper<LongWritable, LongArrayWritable, LongWritable, LongArrayWritable> {

		Random r;
		private String mapTaskId = "";
		JobConf conf;
		int TOURNAMENT_SIZE;
		long[] ind;
		int numKeysProcessed = 0;
		long split = 0;
		@Override
		public void configure(JobConf job) {
			conf = job;
			mapTaskId = job.get("mapred.task.id");
			split = Integer.parseInt(job.get("cga.numSplits"));
			r = new Random(System.nanoTime());
			TOURNAMENT_SIZE = Integer.parseInt(job.get("cga.tournamentSize"));
			ind = new long[TOURNAMENT_SIZE];
			for(int l=0; l<TOURNAMENT_SIZE; l++)
				ind[l] = 0;			
		}		

		public void map(LongWritable key, LongArrayWritable value, OutputCollector<LongWritable, LongArrayWritable> oc, Reporter rep) throws IOException {

			// Generate two individuals from the vector
			LongWritable[][] individuals = new LongWritable[TOURNAMENT_SIZE][LONGS_PER_ARRAY];
			LongWritable[] vector = value.getArray();
			long[] indLong = new long[TOURNAMENT_SIZE];
			for(int l=0; l<TOURNAMENT_SIZE; l++) {
				individuals[l][0] = new LongWritable(l);
				indLong[l] = 0;
				for ( int i=1 ; i< LONGS_PER_ARRAY; i++ ) {
					for ( int j=0; j< LONG_BITS; j++) {
						if(Math.abs(r.nextLong()) <= vector[(i-1)*LONG_BITS + j + 1].get()) {
							indLong[l] |= 1; 
							ind[l]++;
						}

						// Don't shift for the last bit
						if( j != LONG_BITS - 1) {
							indLong[l] = indLong[l] << 1;
						}
					}
					//				System.err.print("Ind1 = " + indLong1 + " Ind2 = " + indLong2 + " ind1 = " + ind1 + " ind2 = " + ind2);
					individuals[l][i] = new LongWritable(indLong[l]);
				}
				oc.collect(key, new LongArrayWritable(individuals[l]));
			}

			oc.collect(key, value);
			numKeysProcessed ++;
			if(numKeysProcessed == split) {
				closeAndWriteOutput();
			}
		}

		public void closeAndWriteOutput() {
			// At the end of Map(), write the number of ones in each individual to a file
			Path tmpDir = new Path( rootDir + "CGA");
			Path outDir = new Path(tmpDir, "global");

			// HDFS does not allow multiple reducers to write to the same file
			Path outFile = new Path(outDir, mapTaskId);
			FileSystem fileSys = null;
			try {
				fileSys = FileSystem.get(conf);
			} catch (IOException e) {
				System.err.println("Error in instantiating FileSystem in mapper close");
				e.printStackTrace();
			}
			SequenceFile.Writer writer = null;
			try {
				writer = SequenceFile.createWriter(fileSys, conf, 
						outFile, LongWritable.class, LongWritable.class, 
						CompressionType.NONE);
			} catch (IOException e) {
				System.err.println("Error in instantiating writer in mapper close");
				e.printStackTrace();
			}
			try {
				for(int l=0; l<TOURNAMENT_SIZE; l++) {
					System.err.println(ind[l] + ":");
					writer.append(new LongWritable(l), new LongWritable(ind[l]));
				}
			} catch (IOException e) {
				System.err.println("Error in appending in mapper close");
				e.printStackTrace();
			}
			try {
				writer.close();
			} catch (IOException e) {
				System.err.println("Error in closing in mapper close");
				e.printStackTrace();
			}
		}
	}

	// Inputs <SplitNum, [1, individual1], [2, individual2] , [3, probabilityVectorSplit]> 
	// Outputs <SplitNum, [3, newProbilityVectorSplit]>
	public static class CGAReducer extends MapReduceBase
	implements Reducer<LongWritable, LongArrayWritable, LongWritable, LongArrayWritable> {

		JobConf conf;
		int winner = -1;
		long n = -1;
		Float uLimit, lLimit;
		private String reduceTaskId = "";
		static boolean firstTime = true;
		int TOURNAMENT_SIZE;
		long[] ind;
		@Override
		public void configure(JobConf job) {
			conf = job;
			n = Long.parseLong(job.get("cga.population"));
			uLimit = Float.parseFloat(job.get("cga.uLimit"));
			lLimit = Float.parseFloat(job.get("cga.lLimit"));
			reduceTaskId = job.get("mapred.task.id");
			TOURNAMENT_SIZE = Integer.parseInt(job.get("cga.tournamentSize"));
			ind = new long[TOURNAMENT_SIZE];
		}	
		int i;
		long winnerI = 0;
		int loser = 0;
		long unConverged = 0;

		int findWinner(Reporter r) {
			// Read global map outputs and figure out the winner
			LongWritable ind1 = new LongWritable();
			LongWritable ind2 = new LongWritable();			
			FileStatus[] fs = null;

			int _winner;
			System.err.println("Starting winner");
			SequenceFile.Reader reader = null;
			FileSystem fileSys2 = null;
			try {
				fileSys2 = FileSystem.get(conf);
				Path tmpDir = new Path( rootDir + "CGA");
				fs = fileSys2.listStatus(new Path(tmpDir, "global"));
			} catch (IOException e) {
				System.out.println("Exception while instantiating reader in find winner");
				e.printStackTrace();
			}
			for(int l=0; l<TOURNAMENT_SIZE; l++) {
				ind[l] = 0;
			}
			System.err.println("Initializing");
			for(int i=0; i<fs.length; i++) {
				try{
					reader = new SequenceFile.Reader(fileSys2, fs[i].getPath(), conf);
				} catch (IOException e) {
					System.out.println("Exception while instantiating reader in find winner");
					e.printStackTrace();
				}
				System.err.println("Reading");
				try {
					while(reader.next(ind1, ind2)) {
						ind[(int)ind1.get()] += ind2.get();
					}
				} catch (IOException e) {
					System.out.println("Exception while reading from reader in find winner");
					e.printStackTrace();
				}
				try {
					reader.close();
				} catch (IOException e) {
					System.out.println("Exception while closing reader in find winner");
					e.printStackTrace();
				}
				System.err.println("CLosing");
				r.progress();
			}
			_winner = -1;
			try {
	 
				for(int l=0; l<TOURNAMENT_SIZE; l++) {
					System.out.println(ind[l]+ ", ");
					if(ind[l] > winnerI) {
						_winner = l;
						winnerI = ind[l];
					}
					if(ind[l] < ind[loser]) {
						loser = l;
					}
				}
			}catch (Exception e) {
				System.err.println("Error in for");
				e.printStackTrace();
			}
			return _winner;		
		}

		public void reduce(LongWritable key, Iterator<LongArrayWritable> values,
				OutputCollector<LongWritable, LongArrayWritable> output, Reporter rep)
		throws IOException {
			i=0;

			if(firstTime) {
				winner = findWinner(rep);
				firstTime = false;
				rep.progress();
			}
			LongWritable[] out = new LongWritable[(LONGS_PER_ARRAY-1)*LONG_BITS + 1];
			out[0] = new LongWritable(TOURNAMENT_SIZE);
			LongWritable[][] arrays = new LongWritable[TOURNAMENT_SIZE + 1][LONGS_PER_ARRAY]; 
			LongWritable[] tmp;
			long delta = Long.MAX_VALUE / n;

			while(values.hasNext()) {
				i++;
				LongArrayWritable lw = values.next();
				tmp = lw.getArray();
				//System.err.println(lw);
				arrays[(int)tmp[0].get()] = tmp.clone();
				if(i==TOURNAMENT_SIZE+1) {
					// Got all values. Proceed to update the vector
					for(int k=1; k<LONGS_PER_ARRAY; k++) {
						long mask = 1;
//						System.err.print("Ind1 = " + new LongArrayWritable(arrays[0]) + " Ind2 = " + new LongArrayWritable(arrays[1]));
						for(int j=0; j<LONG_BITS; j++) {
							long p = arrays[TOURNAMENT_SIZE][k].get();
//							System.err.print(" Prev = " + p);// + " mask = " + mask + " i1 = " + arrays[0][k].get() + " i2 = " + arrays[1][k].get());
							if((arrays[winner][k].get() & mask) != (arrays[loser][k].get() & mask)) {
								if((arrays[winner][k].get() & mask) == mask) {
									p += delta;
								}else {
									p -= delta;
								}
							}
							mask = mask << 1;
	//						System.err.print("New = " + p);
							if(p>= lLimit*Long.MAX_VALUE && p<= uLimit*Long.MAX_VALUE) unConverged++;
							out[(k-1)*LONG_BITS + j + 1] = new LongWritable(p);
						}
					}
					//					System.err.println("\n" + new LongArrayWritable(out));
					output.collect(key, new LongArrayWritable(out));
				} 
			}
		}
		@Override
		public void close() throws IOException {
			// At the end of Map(), write the number of ones in each individual to a file
			Path tmpDir = new Path( rootDir + "CGA");
			Path outDir = new Path(tmpDir, "global-red");

			// HDFS does not allow multiple reducers to write to the same file
			Path outFile = new Path(outDir, reduceTaskId);
			FileSystem fileSys = FileSystem.get(conf);
			SequenceFile.Writer writer = SequenceFile.createWriter(fileSys, conf, 
					outFile, LongWritable.class, LongWritable.class, 
					CompressionType.NONE);
			writer.append(new LongWritable(winnerI), new LongWritable(unConverged));
			writer.close();
		}
	}


	void launch(int numMaps, int numReducers, String jt, String dfs, long strLen, long pop, int iter, int t, int it) {
		long numSplits = 0;
		long times = pop;
		numSplits = (long) Math.ceil(strLen*1.0/numMaps/LONG_BITS/(LONGS_PER_ARRAY-1));
		System.out.println("Splits = " + numSplits);
		long tot = LONG_BITS*(LONGS_PER_ARRAY-1)*numSplits*numMaps;
		pop = ((long) Math.ceil(strLen * Math.log(tot) / Math.log(2))) * times;
		System.out.println("Pop = " + pop);

		while(true) {
			JobConf jobConf = new JobConf(getConf(), CGA.class);
			// turn off speculative execution, because DFS doesn't handle
			// multiple writers to the same file.
			jobConf.setSpeculativeExecution(true);
			jobConf.setInputFormat(SequenceFileInputFormat.class);

			jobConf.set("mapred.map.tasks", "3");
			jobConf.set("mapred.reduce.tasks", "3");
			jobConf.setOutputKeyClass(LongWritable.class);
			jobConf.setOutputValueClass(LongArrayWritable.class);
			jobConf.setOutputFormat(SequenceFileOutputFormat.class);

			jobConf.set("cga.lLimit", "0.1");
			jobConf.set("cga.tournamentSize", "" + t);
			jobConf.set("cga.uLimit", "0.501");
			jobConf.set("cga.continueIterations", "0");
			//			jobConf.set("CGA.vectorSizePerPopulation", strLen/numMaps + "");

			jobConf.setNumReduceTasks(numReducers);

			if (jt != null) { jobConf.set("mapred.job.tracker", jt); }
			if (dfs != null) { FileSystem.setDefaultUri(jobConf, dfs); }
			jobConf.setJobName("CGA-mr-" + it);

			Path tmpDir = new Path(rootDir + "CGA");
			Path inDir1 = new Path(tmpDir, "iter" + it);
			Path outDir1 = new Path(tmpDir, "iter" + (it + 1));
			FileInputFormat.setInputPaths(jobConf, inDir1);
			FileOutputFormat.setOutputPath(jobConf, outDir1);

			FileSystem fileSys = null;
			try {
				fileSys = FileSystem.get(jobConf);
			} catch (IOException e1) {
				e1.printStackTrace();
			}
			
			jobConf.set("cga.numSplits", numSplits + "");
			jobConf.set("cga.population", pop + "");

			if(it == 0) {
				// Initialization
				try {
					fileSys.delete(tmpDir, true);
				} catch(IOException ie) {
					System.out.println("Exception while deleting");
					ie.printStackTrace();
				}
				System.out.println("Deleting dir");

				numSplits = (long) Math.ceil(strLen*1.0/numMaps/LONG_BITS/(LONGS_PER_ARRAY-1));
				System.out.println("Splits = " + numSplits);
				for(int i=0; i < numMaps; i++) {
					Path file = new Path(inDir1, "part-"+String.format("%05d", i));
					SequenceFile.Writer writer = null;
					try {
						writer = SequenceFile.createWriter(fileSys, jobConf, 
								file, LongWritable.class, LongArrayWritable.class, CompressionType.NONE);
					}catch(Exception e) {
						System.out.println("Exception while instantiating writer");
						e.printStackTrace();
					}

					// Generate dummy input					
					LongWritable[] split = new LongWritable[1];
					split[0] = new LongWritable(numSplits);
					try{
						writer.append(new LongWritable(i), new LongArrayWritable(split));
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
				jobConf.setMapperClass(InitialCGAMapper.class);
				jobConf.setReducerClass(IdentityReducer.class);
				jobConf.setNumReduceTasks(0);
			} // End of if it == 0
			else {
				jobConf.setMapperClass(CGAMapper.class);
				jobConf.setReducerClass(CGAReducer.class);
				jobConf.setNumReduceTasks(numReducers);
				try {
					fileSys.delete(outDir1, true);
				} catch(IOException ie) {
					System.out.println("Exception while deleting");
					ie.printStackTrace();
				}
			}

			long startTime = System.currentTimeMillis();

			try {
				JobClient.runJob(jobConf);
			} catch (IOException e) {
				System.out.println("Exception while running job");
				e.printStackTrace();
			}

			// At the end of job, find out the best individual
			if(it > 0) {
				long best = -1;
				Path global = new Path(tmpDir, "global-red");
				
				FileStatus[] fs = null;
				SequenceFile.Reader reader = null;
				try {
					fs = fileSys.listStatus(global);
				} catch (IOException e) {
					System.out.println("Exception while instantiating reader in find winner");
					e.printStackTrace();
				}
				long un = 0;
				for(int i=0; i<fs.length; i++) {
					Path inFile = fs[i].getPath();
					try {
						reader = new SequenceFile.Reader(fileSys, inFile,
								jobConf);
					} catch (IOException e) {
						System.out.println("Exception while instantiating reader");
						e.printStackTrace();
					}
					LongWritable winner1 = new LongWritable(0), winner2 = new LongWritable(0);
					try {
						while(reader.next(winner1, winner2)) {
							if(winner1.get() > best) {
								best = winner1.get();
							}
							un += winner2.get();
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
				// Also delete the tmp dirs
				try {
					fileSys.delete(global, true);
					fileSys.delete(new Path(tmpDir, "global"), true);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				//				System.out.println("CGA:" + it + ":" + LONGS_PER_ARRAY * LONG_BITS + ":" + pop + ":" + finalMax.get() + ":" + (System.currentTimeMillis() - startTime));
				System.out.println("CGA:" + it + ":" + pop + ":" + best + ":" + LONG_BITS*(LONGS_PER_ARRAY-1)*numSplits*numMaps +":" + strLen + ":" + un + ":" + (System.currentTimeMillis() - startTime));
				if(un == 0) break;
			}
			it++;
		}
	}
	/**
	 * Launches all the tasks in order.
	 */
	public int run(String[] args) throws Exception {
		if (args.length != 7) {
			System.err.println("Usage: CGAMR <nMaps> <nReducers> <vectorSize> <nIterations> <popTimesNLogN> <tournamentSize> <iter>");
			ToolRunner.printGenericCommandUsage(System.err);
			return -1;
		}

		int	nMaps = Integer.parseInt(args[0]);
		int	nReducers = Integer.parseInt(args[1]);
		long strLen = Long.parseLong(args[2]);
		int iter = Integer.parseInt(args[3]);
		long pop = Integer.parseInt(args[4]);
		int t =  Integer.parseInt(args[5]);
		int it =  Integer.parseInt(args[6]);
		System.out.println("Number of Maps = " + nMaps);  
		launch(nMaps, nReducers, null, null, strLen, pop, iter, t, it);
		return 0;
	}

	public static void main(String[] argv) throws Exception {
		int res = ToolRunner.run(new Configuration(), new CGA(), argv);
		System.exit(res);
	}
}
