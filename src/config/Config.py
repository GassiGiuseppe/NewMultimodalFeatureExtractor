import os
from src.internal.utils.YamlFileManager import YamlFileManager


class Config:

    def __init__(self, config_file_path):
        """
        Manage the configuration within the config yaml file. This configuration are later needed to understand what
        to do

        Args: config_file_path: it is a string, it can be both absolute path to the file, or relative to the inside
        of the Multimodal-Feature-Extractor folder
        """
        # both absolute and relative path are fine
        self._yaml_manager = YamlFileManager(config_file_path)
        self._data_dict = self._yaml_manager.get_raw_dict()
        self._data_dict = self.__clean_dict(self._data_dict)

    def __clean_dict(self, data):
        """
        It cleans the dict to be easily read in the future.
        It crosses in every element of the dict in search of a list of dict to transform in a big dict:
        if there is a dict, it crosses every value (recalling this method).
        If there is a list, it crosses every item (recalling this method). then if the items are dicts the list
        is swapped with a big dict
        Args:
            data: it's the data contained in the yaml file as a dict

        Returns:
            data: it returns data cleaned, every list of dict is transformed in a single dict

        """
        # using yaml there is a problem:
        # it has no strict rules, so you can have [[{}]] [[]] {[]} {{}} ecc
        # this recursive method transform everything as {...{}...} or {...[]...}
        temp_dict = {}
        if isinstance(data, dict):
            for key in data.keys():
                # the model dict follow a particular configuration that is necessary not to change
                if key != 'model':
                    value = self.__clean_dict(data[key])
                    data.update({key: value})
        if isinstance(data, list):
            for element in data:
                element = self.__clean_dict(element)
                # the following code follow a statement that is always true using yaml:
                # if in the list one element is a dict, so are all the others elements
                if isinstance(element, dict):
                    temp_dict.update(element)
        if bool(temp_dict):
            data = temp_dict
        return data

    def get_gpu(self):
        """

        Returns: the gpu list as a string

        """
        # if there is not a gpu config then "-1" (use cpu only)
        # otherwise return the config
        if 'gpu list' in self._data_dict:
            gpu_list = self._data_dict['gpu list']
            if isinstance(gpu_list, str):
                # es '1' or '1,2'
                return gpu_list
            elif isinstance(gpu_list, int):
                # es 1 -> '1'
                return str(gpu_list)
            elif isinstance(gpu_list, list):
                # es [1,3] -> '1,3'
                return ','.join(str(x) for x in gpu_list)
            else:
                raise SyntaxError('the gpu list is written in a incorrect way')
        else:
            return '-1'

    def has_config(self, origin_of_elaboration, type_of_extraction):
        """
        Search the config in the data dicts then check that this config have values in it
        Args:
            origin_of_elaboration: 'items' or 'interactions'
            type_of_extraction: 'textual', 'visual' or 'audio'

        Returns: Bool True/False if contains the configuration

        """
        if type_of_extraction in self._data_dict and origin_of_elaboration in self._data_dict[type_of_extraction]:
            local_dict = self._data_dict[type_of_extraction][origin_of_elaboration]
            # check if local dict has input/output/model
            if 'input' in local_dict and 'output' in local_dict and 'model' in local_dict:
                # in this case it's all right but must be checked that the values are not empty
                input_value = local_dict['input']
                output_value = local_dict['output']
                model_value = local_dict['model']
                if input_value is not None and output_value is not None and model_value is not None:
                    return True
        return False

    def paths_for_extraction(self, origin_of_elaboration, type_of_extraction):
        """
        Gives the working environments
        Args:
            origin_of_elaboration: 'items' or 'interactions'
            type_of_extraction: 'textual', 'visual' or 'audio'

        Returns: a dict as { 'input_path': input path, 'output_path': output_path }

        """
        # {'input_path': ///, 'output_path': ///}
        relative_input_path = self._data_dict[type_of_extraction][origin_of_elaboration]['input']
        relative_output_path = self._data_dict[type_of_extraction][origin_of_elaboration]['output']

        return {
            'input_path': os.path.join(self._data_dict['dataset'], relative_input_path),
            'output_path': os.path.join(self._data_dict['dataset'], relative_output_path)}

    def get_models_list(self, origin_of_elaboration, type_of_extraction):
        """

        Args:
            origin_of_elaboration: 'items' or 'interactions'
            type_of_extraction: 'textual' or 'visual'

        Returns: a list of the models, every model is a dict with
        'name': the name of the model, in same cases as transformers is repo/model name,
        'output_layers': the layers of extraction,
        'framework': framework to work with tensorflow/torch/transformers
         and a custom flag to manage the preprocessing of the data
        """

        models = self._data_dict[type_of_extraction][origin_of_elaboration]['model']

        for model in models:

            # output_layers has to be a list
            if not isinstance(model['output_layers'], list):
                model.update({'output_layers': [model['output_layers']]})

            # Framework elaboration
            # - if INPUT FRAMEWORK is ['tensorflow', 'torch'] then two different model dicts will be added to the list,
            #   each one identical to the other except for the fact that it contains only one of the 2 type of framework
            #   WARNING: the feature to do both of them in the same model declaration is forbidden since they use
            #   different way to call their layers
            # - if OUTPUT FRAMEWORK is ['tensorflow', 'torch'] then outside of this method it means that
            #   the framework in which operate is not known but only one of them will be executed
            if 'framework' in model.keys():
                framework_value = model['framework']

                if framework_value == ['tensorflow', 'torch']:
                    # this setting does not work properly because the two framework used calls different layers
                    first_model = model
                    first_model.update({'framework': ['tensorflow']})

                    second_model = model
                    second_model.update({'framework': ['torch']})

                    # layers

                    first_model_layers = []
                    second_model_layers = []
                    for layer in model['output_layers']:
                        if isinstance(layer, int):
                            second_model_layers.append(layer)
                        else:
                            first_model_layers.append(layer)

                    first_model.update({'output_layers': first_model_layers})
                    second_model.update({'output_layers': second_model_layers})

                    # models_list.append(second_model)
                    # models_list.append(first_model)

                    # this setting does not work properly because the two framework used calls different layers
                    raise ValueError(' unfortunately calling both framework simultaneity doesnt work')
                # framework value must be a list
                elif isinstance(framework_value, str):
                    model.update({'framework': [framework_value]})

                # the following elif was written with the idea that every type of extraction would have only torch or
                # tensorflow. Now this only make sense in the visual case
                #   elif framework_value != ['tensorflow'] and framework_value != ['torch']:
                #       raise ValueError('the framework tag in the yaml file is not written correctly')
            else:
                # the framework is not set
                if type_of_extraction == 'textual':
                    # textual case
                    # in this case we use the 'transformers' framework
                    model.update({'framework': ['transformers']})
                elif type_of_extraction == 'visual':
                    # it is in the visual case, it uses tensorflow or torch, but doesn't know which one
                    # so both are set as plausible
                    model.update({'framework': ['tensorflow', 'torch']})
                elif type_of_extraction == 'audio':
                    # it is the audio case, it uses torchaudio or transformers
                    # both are plausible, it will try torchaudio and if the model is not in its list, it will try
                    # transformers
                    model.update({'framework': ['torch', 'transformers']})

        return models


