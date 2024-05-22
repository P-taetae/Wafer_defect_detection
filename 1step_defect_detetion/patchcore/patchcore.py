"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import pdb
import patchcore
import patchcore.backbones as backbones
import patchcore.common as common
import patchcore.sampler as sampler

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=sampler.IdentitySampler(),
        nn_method=common.FaissNN(True, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.feature_extractor = torch.hub.load("facebookresearch/dino:main", "dino_vitb8").to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.feature_extractor.eval()
        with torch.no_grad():
            feature = self.feature_extractor.get_intermediate_layers(images)[0]

        # import pdb
        # pdb.set_trace()
        x_norm = feature[:,0,:]
        x_norm = x_norm.squeeze()
        # x_norm = x_norm.unsqueeze(1)
        # x_norm = x_norm.detach().cpu().numpy()

        x_prenorm = feature[:,1:,:]
        # x_prenorm = x_prenorm.detach().cpu().numpy()
        x_prenorm = x_prenorm.squeeze()
        
        x_norm = torch.repeat_interleave(x_norm.unsqueeze(0), x_prenorm.shape[0], dim=0)
        # x_norm = np.repeat(np.expand_dims(x_norm, axis=1), x_prenorm.shape[1], axis=1)
        features = torch.concat([x_norm, x_prenorm], axis=-1)
        features = features.reshape(28, 28, features.shape[-1])
        features = features.unsqueeze(0)
        patch_shapes = [x[1] for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        # features = self.forward_modules["preprocessing"](features)
        # features = self.forward_modules["preadapt_aggregator"](features)

        features = features.reshape(-1, features.shape[-1])

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)
        
        # count = 0
        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image,_ in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)

        features = self.featuresampler.run(features)
        self.anomaly_scorer.fit(detection_features=[features])

    def predict(self, data, is_defect_aug=False):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data,is_defect_aug=is_defect_aug)
        return self._predict(data)

    def _predict_dataloader(self, dataloader, is_defect_aug=False):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        images = []
        count = 0

        if is_defect_aug:
            with tqdm.tqdm(dataloader, desc="Inferring...", leave=True) as data_iterator:
                timings = np.zeros((len(data_iterator), 1))
                rep = 0

                for image,label_gt in data_iterator:
                    #if isinstance(image, dict):
                        #labels_gt.extend(image["is_anomaly"].numpy().tolist())
                        #masks_gt.extend(image["mask"].numpy().tolist())
                    labels_gt.extend(label_gt.numpy().tolist())
                    # masks_gt.extend(mask_gt.numpy().tolist())
                    images.extend(image.numpy())
                        #image = image["image"]

                    _scores, _masks = self._predict(image)
                    #timings[rep] = curr_time

                    for score, mask in zip(_scores, _masks):
                        scores.append(score)
                        masks.append(mask)
                    # if count == 0:
                    #     features = np.expand_dims(_features, axis=0)
                    #     count = count + 1
                    # else:
                    #     features = np.concatenate((features, np.expand_dims(_features, axis=0)), axis=0)
                    #rep = rep + 1
            
            #np.save('ti[]me.npy', timings)
            # print(images[0].shape)
            return scores, masks, labels_gt, images
            # return scores, masks, labels_gt, masks_gt, images
        else:
            with tqdm.tqdm(dataloader, desc="Inferring...", leave=True) as data_iterator:
                timings = np.zeros((len(data_iterator), 1))
                rep = 0

                for image, label_gt, mask_gt in data_iterator:
                    #if isinstance(image, dict):
                        #labels_gt.extend(image["is_anomaly"].numpy().tolist())
                        #masks_gt.extend(image["mask"].numpy().tolist())
                    labels_gt.extend(label_gt.numpy().tolist())
                    masks_gt.extend(mask_gt.numpy().tolist())
                    images.extend(image.numpy())
                        #image = image["image"]
                    _scores, _masks = self._predict(image)
                    #timings[rep] = curr_time

                    for score, mask in zip(_scores, _masks):
                        scores.append(score)
                        masks.append(mask)
                    # if count == 0:
                    #     features = np.expand_dims(_features, axis=0)
                    #     count = count + 1
                    # else:
                    #     features = np.concatenate((features, np.expand_dims(_features, axis=0)), axis=0)
                    #rep = rep + 1

            #np.save('ti[]me.npy', timings)
            print(images[0].shape)
            return scores, masks, labels_gt, masks_gt, images

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            starter.record()

            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]

            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)

            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, 28, 28)

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: common.FaissNN(True, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        #pdb.set_trace()
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
