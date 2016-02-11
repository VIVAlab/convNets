local Flip, parent = torch.class('nn.Flip', 'nn.Module')

-- flip dimensions:
-- n = nn.Flip(k)
-- will flip dim k

function Flip:__init(dimension)
   parent.__init(self)
   self.dimension = dimension
   self.k=torch.LongTensor()
end

function Flip:updateOutput(input)
   self.k=torch.range(input:size(self.dimension),1,-1):type('torch.LongTensor')
   local output = input:index(self.dimension,self.k);
   self.output = self.output:typeAs(output)
   self.output:resizeAs(output):copy(output)
   return self.output
end

function Flip:updateGradInput(input, gradOutput)
   
   gradOutput=gradOutput:index(self.dimension,self.k)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   return self.gradInput
end 