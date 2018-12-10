import fire

class PartialParse(object):
    def __init__(self, sentence):
        self.sentence = sentence
        self.stack = ["ROOT"]
        self.buffer = list(sentence)
        self.dependencies = []
    def parse_step(self, transition):
        if transition == 'S':
            self.stack.append(self.buffer.pop(0))
        elif transition == 'LA':
            self.dependencies.append((self.stack[-1], self.stack[-2]))
            self.stack.pop(-2)
        elif transition == 'RA':
            self.dependencies.append((self.stack[-2],self.stack[-1]))
            self.stack.pop(-1)
    def parse(self, transitions):
        for t in transitions:
            self.parse_step(t)
        return self.dependencies
def minibatch_parse(sentences, model, batch_size):
    partial_parses = [PartialParse(i) for i in sentences]
    unfinished_parses = list(partial_parses)
    dependencies = []
    while len(unfinished_parses):
        minibatch = unfinished_parses[:batch_size]
        transitions = model.predict(minibatch)

        for i in range(len(minibatch)):
            minibatch[i].parse_step(transitions[i])

        unfinished_parses = [i for i in partial_parses if len(i.stack)>1 or len(i.buffer)>0]
         
    dependencies = [i.dependencies for i in partial_parses]

    return dependencies


def test_step(name, transition, stack, buf, deps,
              ex_stack, ex_buf, ex_deps):
    """Tests that a single parse step returns the expected output"""
    pp = PartialParse([])
    pp.stack, pp.buffer, pp.dependencies = stack, buf, deps

    pp.parse_step(transition)
    stack, buf, deps = (tuple(pp.stack), tuple(pp.buffer), tuple(sorted(pp.dependencies)))
    assert stack == ex_stack, \
        "{:} test resulted in stack {:}, expected {:}".format(name, stack, ex_stack)
    assert buf == ex_buf, \
        "{:} test resulted in buffer {:}, expected {:}".format(name, buf, ex_buf)
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)
    print ("{:} test passed!".format(name))


def test_parse_step():
    """Simple tests for the PartialParse.parse_step function
    Warning: these are not exhaustive
    """
    test_step("SHIFT", "S", ["ROOT", "the"], ["cat", "sat"], [],
              ("ROOT", "the", "cat"), ("sat",), ())
    test_step("LEFT-ARC", "LA", ["ROOT", "the", "cat"], ["sat"], [],
              ("ROOT", "cat",), ("sat",), (("cat", "the"),))
    test_step("RIGHT-ARC", "RA", ["ROOT", "run", "fast"], [], [],
              ("ROOT", "run",), (), (("run", "fast"),))


def test_parse():
    """Simple tests for the PartialParse.parse function
    Warning: these are not exhaustive
    """
    sentence = ["parse", "this", "sentence"]
    dependencies = PartialParse(sentence).parse(["S", "S", "S", "LA", "RA", "RA"])
    dependencies = tuple(sorted(dependencies))
    expected = (('ROOT', 'parse'), ('parse', 'sentence'), ('sentence', 'this'))
    assert dependencies == expected,  \
        "parse test resulted in dependencies {:}, expected {:}".format(dependencies, expected)
    assert tuple(sentence) == ("parse", "this", "sentence"), \
        "parse test failed: the input sentence should not be modified"
    print ("parse test passed!")


class DummyModel:
    """Dummy model for testing the minibatch_parse function
    First shifts everything onto the stack and then does exclusively right arcs if the first word of
    the sentence is "right", "left" if otherwise.
    """
    def predict(self, partial_parses):
        return [("RA" if pp.stack[1] is "right" else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]


def test_dependencies(name, deps, ex_deps):
    """Tests the provided dependencies match the expected dependencies"""
    deps = tuple(sorted(deps))
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)


def test_minibatch_parse():
    """Simple tests for the minibatch_parse function
    Warning: these are not exhaustive
    """
    sentences = [["right", "arcs", "only"],
                 ["right", "arcs", "only", "again"],
                 ["left", "arcs", "only"],
                 ["left", "arcs", "only", "again"]]
    deps = minibatch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'right'), ('arcs', 'only'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[1],
                      (('ROOT', 'right'), ('arcs', 'only'), ('only', 'again'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[2],
                      (('only', 'ROOT'), ('only', 'arcs'), ('only', 'left')))
    test_dependencies("minibatch_parse", deps[3],
                      (('again', 'ROOT'), ('again', 'arcs'), ('again', 'left'), ('again', 'only')))
    print ("minibatch_parse test passed!")
if __name__ == '__main__':
    # fire.Fire()
    test_parse_step()
    test_parse()
    test_minibatch_parse()