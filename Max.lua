local Max, parent = torch.class('nn.Max', 'nn.Module')

function Max:__init(dimension)
   parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
   self.indices = torch.LongTensor()
end

function Max:updateOutput(input)
   torch.max(self.output, self.indices, input, self.dimension)
   if self.output:dim() > 1 then
      self.output = self.output:select(self.dimension, 1)
   end
   return self.output
end

function Max:updateGradInput(input, gradOutput)
   gradOutput = gradOutput:viewAs(self.indices)
   self.gradInput:resizeAs(input):zero()
   self.gradInput:indexedWrite(self.dimension, self.indices, gradOutput)
   return self.gradInput
end
