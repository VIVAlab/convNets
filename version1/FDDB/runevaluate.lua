folders = {'fold-01','fold-02','fold-03','fold-04','fold-01','fold-05','fold-06','fold-07','fold-08','fold-09','fold-10'}
logFile = io.open('test.log','w+')
for i=1,#folders do
	folder = folders[i]
	dofile 'evaluate.lua'
end

os.execute('../FDDB/runEvaluate.pl')
