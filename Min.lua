local Min, parent = torch.class('nn.Min', 'nn.Module')

function Min:__init(dimension)
   parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
   self.indices = torch.LongTensor()
end

function Min:updateOutput(input)
   torch.min(self.output, self.indices, input, self.dimension)
   if self.output:dim() > 1 then
      self.output = self.output:select(self.dimension, 1)
   end
   return self.output
end

function Min:updateGradInput(input, gradOutput)
   gradOutput = gradOutput:viewAs(self.indices)
   self.gradInput:resizeAs(input):zero()
   self.gradInput:indexedWrite(self.dimension, self.indices, gradOutput)
   return self.gradInput
end
