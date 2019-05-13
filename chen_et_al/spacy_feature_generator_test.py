import unittest

from chen_et_al.spacy_feature_generator import create_spacy_feature_generator


def tf_supplier(x):
    if x == "It":
        return 5
    if x == "happens":
        return 10
    if x == "John":
        return 8
    return None


def word_embedding_vector_generator(x):
    if x == "It":
        return [101, 102, 103]
    if x == "happens":
        return [505, 506, 507]
    if x == "John":
        return [401, 402, 403]
    return None


class SpacyFeatureGeneratorTest(unittest.TestCase):

    def setUp(self):
        self.gen = create_spacy_feature_generator(word_embedding_vector_generator, tf_supplier)

    def test_size(self):
        self.assertListEqual(self.gen.generate_feature_vector_for_paragraph(["It", "happens", "John"], ["What", "happened", "it"]),
                             [[([101, 102, 103], [False, True, True], [5, 'PRP', False])],
                              [([505, 506, 507], [False, False, True], [10, 'VBZ', False])],
                              [([401, 402, 403], [False, False, False], [8, 'NNP', True])]]
                             )
