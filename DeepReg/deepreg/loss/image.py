"""Provide different loss or metrics classes for images."""
import tensorflow as tf
from tensorflow.python.ops import array_ops

from deepreg.loss.util import NegativeLossMixin
from deepreg.loss.util import gaussian_kernel1d_size as gaussian_kernel1d
from deepreg.loss.util import (
    rectangular_kernel1d,
    separable_filter,
    triangular_kernel1d,
)
from deepreg.registry import REGISTRY

EPS = tf.keras.backend.epsilon()


@REGISTRY.register_loss(name="ssd")
class SumSquaredDifference(tf.keras.losses.Loss):
    """
    Sum of squared distance between y_true and y_pred.

    y_true and y_pred have to be at least 1d tensor, including batch axis.
    """

    def __init__(
        self,
        reduction: str = tf.keras.losses.Reduction.SUM,
        name: str = "SumSquaredDifference",
    ):
        """
        Init.

        :param reduction: using SUM reduction over batch axis,
            calling the loss like `loss(y_true, y_pred)` will return a scalar tensor.
        :param name: name of the loss
        """
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch,)
        """
        loss = tf.math.squared_difference(y_true, y_pred)
        loss = tf.keras.layers.Flatten()(loss)
        return tf.reduce_mean(loss, axis=1)


class GlobalMutualInformation(tf.keras.losses.Loss):
    """
    Differentiable global mutual information via Parzen windowing method.

    y_true and y_pred have to be at least 4d tensor, including batch axis.

    Reference: https://dspace.mit.edu/handle/1721.1/123142,
        Section 3.1, equation 3.1-3.5, Algorithm 1
    """

    def __init__(
        self,
        num_bins: int = 23,
        sigma_ratio: float = 0.5,
        reduction: str = tf.keras.losses.Reduction.SUM,
        name: str = "GlobalMutualInformation",
    ):
        """
        Init.

        :param num_bins: number of bins for intensity, the default value is empirical.
        :param sigma_ratio: a hyper param for gaussian function
        :param reduction: using SUM reduction over batch axis,
            calling the loss like `loss(y_true, y_pred)` will return a scalar tensor.
        :param name: name of the loss
        """
        super().__init__(reduction=reduction, name=name)
        self.num_bins = num_bins
        self.sigma_ratio = sigma_ratio

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, dim1, dim2, dim3)
            or (batch, dim1, dim2, dim3, ch)
        :param y_pred: shape = (batch, dim1, dim2, dim3)
            or (batch, dim1, dim2, dim3, ch)
        :return: shape = (batch,)
        """
        # adjust
        if len(y_true.shape) == 4:
            y_true = tf.expand_dims(y_true, axis=4)
            y_pred = tf.expand_dims(y_pred, axis=4)
        assert len(y_true.shape) == len(y_pred.shape) == 5

        # intensity is split into bins between 0, 1
        y_true = tf.clip_by_value(y_true, 0, 1)
        y_pred = tf.clip_by_value(y_pred, 0, 1)
        bin_centers = tf.linspace(0.0, 1.0, self.num_bins)  # (num_bins,)
        bin_centers = tf.cast(bin_centers, dtype=y_true.dtype)
        bin_centers = bin_centers[None, None, ...]  # (1, 1, num_bins)
        sigma = (
            tf.reduce_mean(bin_centers[:, :, 1:] - bin_centers[:, :, :-1])
            * self.sigma_ratio
        )  # scalar, sigma in the Gaussian function (weighting function W)
        preterm = 1 / (2 * tf.math.square(sigma))  # scalar
        batch, w, h, z, c = y_true.shape
        y_true = tf.reshape(y_true, [batch, w * h * z * c, 1])  # (batch, nb_voxels, 1)
        y_pred = tf.reshape(y_pred, [batch, w * h * z * c, 1])  # (batch, nb_voxels, 1)
        nb_voxels = y_true.shape[1] * 1.0  # w * h * z, number of voxels

        # each voxel contributes continuously to a range of histogram bin
        ia = tf.math.exp(
            -preterm * tf.math.square(y_true - bin_centers)
        )  # (batch, nb_voxels, num_bins)
        ia /= tf.reduce_sum(ia, -1, keepdims=True)  # (batch, nb_voxels, num_bins)
        ia = tf.transpose(ia, (0, 2, 1))  # (batch, num_bins, nb_voxels)
        pa = tf.reduce_mean(ia, axis=-1, keepdims=True)  # (batch, num_bins, 1)

        ib = tf.math.exp(
            -preterm * tf.math.square(y_pred - bin_centers)
        )  # (batch, nb_voxels, num_bins)
        ib /= tf.reduce_sum(ib, -1, keepdims=True)  # (batch, nb_voxels, num_bins)
        pb = tf.reduce_mean(ib, axis=1, keepdims=True)  # (batch, 1, num_bins)

        papb = tf.matmul(pa, pb)  # (batch, num_bins, num_bins)
        pab = tf.matmul(ia, ib)  # (batch, num_bins, num_bins)
        pab /= nb_voxels

        # MI: sum(P_ab * log(P_ab/P_ap_b))
        div = (pab + EPS) / (papb + EPS)
        return tf.reduce_sum(pab * tf.math.log(div + EPS), axis=[1, 2])

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["num_bins"] = self.num_bins
        config["sigma_ratio"] = self.sigma_ratio
        return config


@REGISTRY.register_loss(name="gmi")
class GlobalMutualInformationLoss(NegativeLossMixin, GlobalMutualInformation):
    """Revert the sign of GlobalMutualInformation."""


class LocalNormalizedCrossCorrelation(tf.keras.losses.Loss):
    """
    Local squared zero-normalized cross-correlation.

    Denote y_true as t and y_pred as p. Consider a window having n elements.
    Each position in the window corresponds a weight w_i for i=1:n.

    Define the discrete expectation in the window E[t] as

        E[t] = sum_i(w_i * t_i) / sum_i(w_i)

    Similarly, the discrete variance in the window V[t] is

        V[t] = E[t**2] - E[t] ** 2

    The local squared zero-normalized cross-correlation is therefore

        E[ (t-E[t]) * (p-E[p]) ] ** 2 / V[t] / V[p]

    where the expectation in numerator is

        E[ (t-E[t]) * (p-E[p]) ] = E[t * p] - E[t] * E[p]

    Different kernel corresponds to different weights.

    For now, y_true and y_pred have to be at least 4d tensor, including batch axis.

    Reference:

        - Zero-normalized cross-correlation (ZNCC):
            https://en.wikipedia.org/wiki/Cross-correlation
        - Code: https://github.com/voxelmorph/voxelmorph/blob/legacy/src/losses.py
    """

    kernel_fn_dict = dict(
        gaussian=gaussian_kernel1d,
        rectangular=rectangular_kernel1d,
        triangular=triangular_kernel1d,
    )

    def __init__(
        self,
        kernel_size: int = 9,
        kernel_type: str = "rectangular",
        reduction: str = tf.keras.losses.Reduction.SUM,
        name: str = "LocalNormalizedCrossCorrelation",
    ):
        """
        Init.

        :param kernel_size: int. Kernel size or kernel sigma for kernel_type='gauss'.
        :param kernel_type: str, rectangular, triangular or gaussian
        :param reduction: using SUM reduction over batch axis,
            calling the loss like `loss(y_true, y_pred)` will return a scalar tensor.
        :param name: name of the loss
        """
        super().__init__(reduction=reduction, name=name)
        if kernel_type not in self.kernel_fn_dict.keys():
            raise ValueError(
                f"Wrong kernel_type {kernel_type} for LNCC loss type. "
                f"Feasible values are {self.kernel_fn_dict.keys()}"
            )
        self.kernel_fn = self.kernel_fn_dict[kernel_type]
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size

        # (kernel_size, )
        self.kernel = self.kernel_fn(kernel_size=self.kernel_size)
        # E[1] = sum_i(w_i), ()
        self.kernel_vol = tf.reduce_sum(
            self.kernel[:, None, None]
            * self.kernel[None, :, None]
            * self.kernel[None, None, :]
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, dim1, dim2, dim3)
            or (batch, dim1, dim2, dim3, ch)
        :param y_pred: shape = (batch, dim1, dim2, dim3)
            or (batch, dim1, dim2, dim3, ch)
        :return: shape = (batch,)
        """
        # adjust
        if len(y_true.shape) == 4:
            y_true = tf.expand_dims(y_true, axis=4)
            y_pred = tf.expand_dims(y_pred, axis=4)
        assert len(y_true.shape) == len(y_pred.shape) == 5

        # t = y_true, p = y_pred
        # (batch, dim1, dim2, dim3, ch)
        t2 = y_true * y_true
        p2 = y_pred * y_pred
        tp = y_true * y_pred

        # sum over kernel
        # (batch, dim1, dim2, dim3, 1)
        t_sum = separable_filter(y_true, kernel=self.kernel)  # E[t] * E[1]
        p_sum = separable_filter(y_pred, kernel=self.kernel)  # E[p] * E[1]
        t2_sum = separable_filter(t2, kernel=self.kernel)  # E[tt] * E[1]
        p2_sum = separable_filter(p2, kernel=self.kernel)  # E[pp] * E[1]
        tp_sum = separable_filter(tp, kernel=self.kernel)  # E[tp] * E[1]

        # average over kernel
        # (batch, dim1, dim2, dim3, 1)
        t_avg = t_sum / self.kernel_vol  # E[t]
        p_avg = p_sum / self.kernel_vol  # E[p]

        # shape = (batch, dim1, dim2, dim3, 1)
        cross = tp_sum - p_avg * t_sum  # E[tp] * E[1] - E[p] * E[t] * E[1]
        t_var = t2_sum - t_avg * t_sum  # V[t] * E[1]
        p_var = p2_sum - p_avg * p_sum  # V[p] * E[1]

        # (E[tp] - E[p] * E[t]) ** 2 / V[t] / V[p]
        ncc = (cross * cross + EPS) / (t_var * p_var + EPS)

        return tf.reduce_mean(ncc, axis=[1, 2, 3, 4])

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["kernel_size"] = self.kernel_size
        config["kernel_type"] = self.kernel_type
        return config


@REGISTRY.register_loss(name="lncc")
class LocalNormalizedCrossCorrelationLoss(
    NegativeLossMixin, LocalNormalizedCrossCorrelation
):
    """Revert the sign of LocalNormalizedCrossCorrelation."""


class GlobalNormalizedCrossCorrelation(tf.keras.losses.Loss):
    """
    Global squared zero-normalized cross-correlation.

    Compute the squared cross-correlation between the reference and moving images
    y_true and y_pred have to be at least 4d tensor, including batch axis.

    Reference:

        - Zero-normalized cross-correlation (ZNCC):
            https://en.wikipedia.org/wiki/Cross-correlation

    """

    def __init__(
        self,
        reduction: str = tf.keras.losses.Reduction.AUTO,
        name: str = "GlobalNormalizedCrossCorrelation",
    ):
        """
        Init.
        :param reduction: using AUTO reduction,
            calling the loss like `loss(y_true, y_pred)` will return a scalar tensor.
        :param name: name of the loss
        """
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch,)
        """

        axis = [a for a in range(1, len(y_true.shape))]
        mu_pred = tf.reduce_mean(y_pred, axis=axis, keepdims=True)
        mu_true = tf.reduce_mean(y_true, axis=axis, keepdims=True)
        var_pred = tf.math.reduce_variance(y_pred, axis=axis)
        var_true = tf.math.reduce_variance(y_true, axis=axis)
        numerator = tf.abs(
            tf.reduce_mean((y_pred - mu_pred) * (y_true - mu_true), axis=axis)
        )

        return (numerator * numerator + EPS) / (var_pred * var_true + EPS)


@REGISTRY.register_loss(name="gncc")
class GlobalNormalizedCrossCorrelationLoss(
    NegativeLossMixin, GlobalNormalizedCrossCorrelation
):
    """Revert the sign of GlobalNormalizedCrossCorrelation."""


class LinearCorrelationOfLinearCombination(tf.keras.losses.Loss):
    """
    LinearCorrelationOfLinearCombination. Non-differentiable, to be used with BOBYQA.

    y_true and y_pred have to be at least 4d tensor, including batch axis.

    Reference:
    - Paper: https://doi.org/10.1016/j.media.2014.04.008
    - Code: http://campar.in.tum.de/Main/LC2Code (matlab)
    """

    def __init__(
        self,
        reduction: str = tf.keras.losses.Reduction.SUM,
        name: str = "LinearCorrelationOfLinearCombination",
        patch: bool = True,
        patch_size: int = 7,
        neighborhood: bool = False
    ):
        """
        Init.

        :param reduction: using SUM reduction over batch axis,
            calling the loss like `loss(y_true, y_pred)` will return a scalar tensor.
        :param name: name of the loss
        :param patch: whether to use patches (bool)
        :param patch_size: default patch size [7 from authors] (int)
        :param neighborhood: use avg neighborhood value for fitting LC2
        """
        self.neighborhood = neighborhood
        self.patch = patch
        self.patch_size = patch_size
        super().__init__(reduction=reduction, name=name)

    def intensityTransform(self, y_true, y_pred, y_pred_full=None, mask=None):
        '''
        Find intensity mapping rom y_pred to y_true using
        linear regression (LSTSQ)
            :param y_true: vector - US
                shape: (n) or (n, 1)
            :param y_pred: data matrix - MRI intensity & gradient
                shape: (n, ch)

            :return: estimate of y_true from y_pred
                shape: (n)
        '''
        if len(y_true.shape) == 1:
            y_true = tf.expand_dims(y_true, axis=1)
        N, C = y_pred.shape

        # X = [A B 1] Where A is intensity, B is gradient mag, and 1 is ones
        y_pred_feat = tf.concat([y_pred, tf.ones([N, 1])], 1)

        # if y_pred_full is None:
        #     y_pred_feat = tf.concat([X, tf.ones([N, 1])], 1)
        # else:
        #     # if using the neighborhood, create vectors for the 4 pixels surrounding a pixel + their 4 gradients
        #     # TODO

        #     # Currently, takes average value of intensity and average value of gradient neighborhoods
        #     k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4
        #     intsty_nbhd = correlate2d(X_full[:, :, 0], k, mode='same', boundary='symm')
        #     intsty_nbhd = np.expand_dims(intsty_nbhd.reshape(-1)[mask], 1)
        #     grad_nbhd = correlate2d(X_full[:, :, 1], k, mode='same', boundary='symm').reshape(-1)
        #     grad_nbhd = np.expand_dims(grad_nbhd.reshape(-1)[mask], 1)
        #     y_pred_feat = np.concatenate((X, intsty_nbhd, grad_nbhd, np.ones((N, 1))), axis=1)

        # w = pinv(X)*y  --->  y_hat = Xw
        params = tf.linalg.pinv(y_pred_feat) @ y_true
        y_true_hat = tf.squeeze(y_pred_feat @ params)
        return y_true_hat

    # LC2 Code:
    def lc2Similarity(self, y_true, y_pred, extend=False):
        '''
        Return Linear Correlation of Linear Combination (LC2)
        similarity measure for two input images.
        :param y_true: one channel image of size (h, w, d) or (h, w, d, ch=1) (US)
        :param y_pred: multi-channel image of size (h, w, d, ch=2) (MRI)
            y_pred[:, :, :, 0] = image magnitude
            y_pred[:, :, :, 1] = image gradient

        :returns: scalar similarity measure
        '''

        # Define similarity value
        similarity = -1
        weight = 0
        measure = 0

        y_true = tf.reshape(y_true, [-1])
        # y_pred_orig = y_pred
        y_pred = tf.reshape(y_pred, [-1, y_pred.shape[3]])

        num_pixels = y_pred.shape[0]
        mask = y_true > 0
        num_nonzero = tf.reduce_sum(tf.cast(mask, dtype=tf.float32))

        y_true_vec = y_true[mask]
        y_pred_vec = y_pred[mask]
        _, var_y_true_vec = tf.nn.moments(y_true_vec, [0])

        if var_y_true_vec > 1e-12:  # if variance is nonzero
            if num_nonzero > num_pixels / 2:  # if more than half the pixels aren't zero
                y_true_hat = self.intensityTransform(y_true_vec, y_pred_vec)
                # if extend:
                #     y_true_hat = self.intensityTransform(y_true_vec, y_pred_vec, y_pred_full=y_pred_orig, mask=mask)

                # DAK DEBUG
                # mse = tf.math.reduce_mean(tf.math.pow(y_true_vec - y_true_hat, 2))

                _, var_y_diff_vec = tf.nn.moments(y_true_vec - y_true_hat, [0])
                similarity = 1 - (var_y_diff_vec / var_y_true_vec)
                weight = tf.math.sqrt(var_y_true_vec)
                measure = weight * similarity

        if (similarity == -1):
            similarity = 0
            weight = 0
            measure = 0

        result = tf.concat([similarity, weight, measure], axis=0)

        return result

    def gradient_mag(self, image: tf.Tensor) -> tf.Tensor:
        """
        Return gradient magnitude for each pixel, calculated as
        second difference internally, and first difference along
        borders. Same as MATLAB.

        :param image: input image of dimension 5
            shape = (batch, dim1, dim2, dim3, ch)
        :return: shape = (batch, dim1, dim2, dim3, ch)
        """
        assert len(image.shape) == 5
        image_shape = array_ops.shape(image)

        dy_up = tf.expand_dims(image[:, 1, :, :, :] - image[:, 0, :, :, :], axis=1)
        dy = (image[:, 2:, :, :, :] - image[:, :-2, :, :, :]) / 2
        dy_down = tf.expand_dims(image[:, -2, :, :, :] - image[:, -1, :, :, :], axis=1)
        dy = array_ops.concat([dy_up, dy, dy_down], 1)
        dy = array_ops.reshape(dy, image_shape)

        dx_left = tf.expand_dims(image[:, :, 1, :, :] - image[:, :, 0, :, :], axis=2)
        dx = (image[:, :, 2:, :, :] - image[:, :, :-2, :, :]) / 2
        dx_right = tf.expand_dims(image[:, :, -2, :, :] - image[:, :, -1, :, :], axis=2)
        dx = array_ops.concat([dx_left, dx, dx_right], 2)
        dx = array_ops.reshape(dx, image_shape)

        dz_in = tf.expand_dims(image[:, :, :, 1, :] - image[:, :, :, 0, :], axis=3)
        dz = (image[:, :, :, 2:, :] - image[:, :, :, :-2, :]) / 2
        dz_out = tf.expand_dims(image[:, :, :, -2, :] - image[:, :, :, -1, :], axis=3)
        dz = array_ops.concat([dz_in, dz, dz_out], 3)
        dz = array_ops.reshape(dz, image_shape)

        magnitude = tf.math.sqrt(tf.math.pow(dx, 2) + tf.math.pow(dy, 2) + tf.math.pow(dz, 2))
        return magnitude  # (batch, h, w, d)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch. batch must be 1.

        :param y_true: Fixed image = US
            shape = (batch, dim1, dim2, dim3)
                or (batch, dim1, dim2, dim3, ch)
        :param y_pred: Moving image = MRI
            shape = (batch, dim1, dim2, dim3)
                or (batch, dim1, dim2, dim3, ch)
        :return: shape = (batch,)
        """
        # adjust
        if len(y_true.shape) == 4:
            y_true = tf.expand_dims(y_true, axis=4)
            y_pred = tf.expand_dims(y_pred, axis=4)
        assert len(y_true.shape) == len(y_pred.shape) == 5

        y_pred_mag = self.gradient_mag(y_pred)
        y_pred_cat = tf.concat([y_pred, y_pred_mag], 4)

        if self.patch:
            sizes = [1, self.patch_size, self.patch_size, self.patch_size, 1]
            strides = [1, 1, 1, 1, 1]
            padding = 'SAME'  # zero pads. LC2 doesn't consider 0 values, so no artifacts.
            y_true_patches = tf.extract_volume_patches(y_true, sizes, strides, padding)
            y_pred_patches = tf.extract_volume_patches(y_pred_cat, sizes, strides, padding)

            # patches come interlaced by channel: image mag, grad mag, etc.
            num_patches = y_true_patches.shape[4]
            lc2_result = tf.zeros([num_patches, 3])  # create array for return values
            for i, patch in enumerate(tf.unstack(y_true_patches, axis=4)):
                lc2_res_i = self.lc2Similarity(y_true_patches[0, :, :, :, i], y_pred_patches[0, :, :, :, 2 * i:(2 * i) + 2])
                lc2_result = tf.tensor_scatter_nd_update(lc2_result, [[i]], [lc2_res_i])
            similarity = tf.reduce_sum(lc2_result[:, 2]) / tf.reduce_sum(lc2_result[:, 1])
            similarity = tf.expand_dims(similarity, axis=0)
        else:
            lc2_result = self.lc2Similarity(y_true, y_pred_cat)
            similarity = lc2_result[:, 0]  # (batch,)
        return similarity

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["neighborhood"] = self.neighborhood
        config["patch"] = self.patch
        config["patch_size"] = self.patch_size
        return config


@REGISTRY.register_loss(name="lc2")
class LinearCorrelationOfLinearCombinationLoss(NegativeLossMixin, LinearCorrelationOfLinearCombination):
    """Revert the sign of LinearCorrelationOfLinearCombination."""
