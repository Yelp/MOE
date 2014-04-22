# -*- coding: utf-8 -*-

import numpy # for sci comp
import matplotlib.pylab as plt # for plotting
from matplotlib.backends.backend_pdf import PdfPages # for exporting to pdf
import time # for timing

from src.python.models.optimal_gaussian_process_linked_cpp import OptimalGaussianProcessLinkedCpp


class PlottableOptimalGaussianProcess(OptimalGaussianProcessLinkedCpp):
    """docstring!
    """

    def plot_contour_and_quiver(self, resolution=20, points_being_sampled=[], save_figure=False, figure_path=None):
        """Plot the GPP contour plot and grad_mu"""

        if not len(self.domain) == 1:
            raise(ValueError, "Can only plot contour and quiver for 1D functions")

        plotting_points = numpy.arange(self.domain[0][0], self.domain[0][1], (self.domain[0][1] - self.domain[0][0])/float(resolution))

        sample_grad_EI_dx1 = numpy.zeros((resolution, resolution))
        sample_grad_EI_dx2 = numpy.zeros((resolution, resolution))
        sample_EI = numpy.zeros((resolution, resolution))

        for i, x1 in enumerate(plotting_points):
            for j, x2 in enumerate(plotting_points):

                union_of_points = [numpy.array([x1]), numpy.array([x2])]
                union_of_points.extend(points_being_sampled)

                union_of_points_without_x1 = [numpy.array([x2])]
                union_of_points_without_x1.extend(points_being_sampled)

                union_of_points_without_x2 = [numpy.array([x1])]
                union_of_points_without_x2.extend(points_being_sampled)

                sample_grad_EI_dx1[i][j] = self.get_expected_grad_EI(numpy.array([x1]), union_of_points_without_x1)
                sample_grad_EI_dx2[i][j] = self.get_expected_grad_EI(numpy.array([x2]), union_of_points_without_x2)
                sample_EI[i][j] = self.get_expected_improvement(union_of_points)
                print x1,x2

        X, Y = numpy.meshgrid(plotting_points, plotting_points)

        if not passed_fig:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = passed_fig
            ax = fig.add_subplot(fig_place)

        if points_being_sampled:
            ax.plot([numpy.min(plotting_points), numpy.max(plotting_points)], [points_being_sampled[0], points_being_sampled[0]], 'b')
            if len(points_being_sampled) == 2:
                ax.plot([numpy.min(plotting_points), numpy.max(plotting_points)], [points_being_sampled[1], points_being_sampled[1]], 'b')



        Q = plt.quiver(Y, X, sample_grad_EI_dx1, sample_grad_EI_dx2)
        CS = plt.contour(X, Y, sample_EI)
        ax.set_title("Sample EI and grad EI")
        ax.set_xlim(numpy.min(plotting_points), numpy.max(plotting_points))
        ax.set_ylim(numpy.min(plotting_points), numpy.max(plotting_points))

        if save_figure:
            if not pdf_stream:
                #raise(ValueError, "If save_figure is True a pdf_stream must me specified")
                return fig
            else:
                pdf_stream.savefig()
        else:
            plt.show()

def plot_1D_EI_during_sample(GPP, plotting_points=numpy.arange(0,3.01,0.01), points_being_sampled=None, next_points_to_be_sampled=None, true_function=None, save_figure=False, pdf_stream=None, passed_fig=None, fig_place=223):
    if not passed_fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = passed_fig
        ax = fig.add_subplot(fig_place)

    if true_function:
        true_values = numpy.zeros(len(plotting_points))
        for i, point in enumerate(plotting_points):
            true_values[i] = true_function(point)
        ax.plot(plotting_points, GPP.best_so_far - true_values, 'g--')

    EI_values = numpy.zeros(len(plotting_points))

    for i, point in enumerate(plotting_points):
        union_of_points = [point]
        if points_being_sampled:
            union_of_points.extend(points_being_sampled)
        EI_values[i] = numpy.max([0.0, GPP.get_expected_improvement(union_of_points, iterations=1000) - GPP.get_expected_improvement(points_being_sampled, iterations=1000)])

    ax.plot(domain, EI_values)

    if points_being_sampled:
        y_vals = numpy.zeros(len(points_being_sampled))
        ax.plot(points_being_sampled, y_vals, 'b*')

    if next_points_to_be_sampled:
        for i, next_point in enumerate(next_points_to_be_sampled):
            if i == 0:
                ax.plot([next_point, next_point], [0, numpy.max(EI_values)], 'r-')
            else:
                ax.plot([next_point, next_point], [0, numpy.max(EI_values)], 'g-')


    ax.set_xlim(numpy.min(domain), numpy.max(domain))

    if save_figure:
        if not pdf_stream:
            #raise(ValueError, "If save_figure is True a pdf_stream must me specified")
            return fig
        else:
            pdf_stream.savefig()
    else:
        plt.show()


def plot_1D_EI(GPP, domain=numpy.arange(0,3.01,0.01), true_function=None, save_figure=False, pdf_stream=None, passed_fig=None, fig_place=223):

    if not passed_fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = passed_fig
        ax = fig.add_subplot(fig_place)

    if true_function:
        true_values = numpy.zeros(len(domain))
        for i, point in enumerate(domain):
            true_values[i] = true_function(point)
        ax.plot(domain, GPP.best_so_far - true_values, 'g--')

    EI_values = numpy.zeros(len(domain))
    EI_analytic = numpy.zeros(len(domain))
    EI_deriv = numpy.zeros(len(domain))

    for i, point in enumerate(domain):
        #EI_values[i] = GPP.get_expected_improvement([point], iterations=1000)
        EI_analytic[i] = GPP.get_1D_analytic_expected_improvement(numpy.array([point]))
        #EI_deriv[i] = GPP.get_1D_analytic_grad_EI(point)

    #ax.plot(domain, EI_values)
    ax.plot(domain, EI_analytic)
    #ax.plot(domain, EI_deriv)
    ax.set_xlim(numpy.min(domain), numpy.max(domain))
    ax.set_title("1-D Expected Improvement")

    if save_figure:
        if not pdf_stream:
            #raise(ValueError, "If save_figure is True a pdf_stream must me specified")
            return fig
        else:
            pdf_stream.savefig()
    else:
        plt.show()

def plot_GPP(GPP, domain=None, points_being_sampled=None, true_function=None, save_figure=False, pdf_stream=None, passed_fig=None, fig_place=221):
    """Plot the GPP as it currently over some domain with a potential true_function for comparison"""

    if domain==None:
        domain=numpy.arange(0,3.01,0.01)

    if not passed_fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = passed_fig
        ax = fig.add_subplot(fig_place)

    if true_function:
        true_values = numpy.zeros(len(domain))
        for i, point in enumerate(domain):
            true_values[i] = true_function(point)
        ax.plot(domain, true_values)
    else:
        true_values = None

    if points_being_sampled:
        y_vals, y_vars = GPP.get_mean_and_var_of_points(points_being_sampled)
        ax.plot(points_being_sampled, y_vals, 'b*')


    massaged_domain = []
    for point in domain:
        massaged_domain.append(numpy.array([point]))
    mu_star, var_star = GPP.get_mean_and_var_of_points(massaged_domain)

    # TODO: make pretty/pythonic
    pointwise_var = numpy.zeros(len(domain))
    for i in range(len(domain)):
        pointwise_var[i] = var_star[i][i]

    # plot mean and var accross domain
    ax.plot(domain, mu_star, '--')
    ax.fill_between(domain, mu_star - pointwise_var, mu_star + pointwise_var, facecolor='green', alpha=0.2)

    # show sampled points with red x's
    sampled_x_points = numpy.zeros(len(GPP.points_sampled))
    sampled_x_vals = numpy.zeros(len(GPP.points_sampled))
    for i, sample in enumerate(GPP.points_sampled):
        sampled_x_points[i] = sample.point
        sampled_x_vals[i] = sample.value
    plt.plot(sampled_x_points, sampled_x_vals, 'rx')

    ax.set_title("GPP of points sampled")

    # set limits
    y_min = numpy.min([numpy.min(mu_star - pointwise_var), numpy.min(true_values)])
    y_max = numpy.max([numpy.max(mu_star - pointwise_var), numpy.max(true_values)])
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(numpy.min(domain), numpy.max(domain))

    if save_figure:
        if not pdf_stream:
            #raise(ValueError, "If save_figure is True a pdf_stream must me specified")
            return fig
        else:
            pdf_stream.savefig()
    else:
        plt.show()

def full_stepper_plot(number_of_steps=10):
    domain = numpy.arange(-1,1.01,0.01)
    def true_function(x):
        return 4*(x - 0.45)**2

    covariance_of_process = gp.CovarianceOfProcess(l=0.1, alpha=10.0)

    # kind of hacky to get initialized (initial_best_so_far)
    GPP = gp.GaussianProcess(covariance_of_process=covariance_of_process, initial_best_so_far=true_function(-2))

    pdf_stream = PdfPages("full_stepper_plot.pdf")

    # initialize the GPP
    point = 0.0
    p = SamplePoint(point, true_function(point))
    #GPP.add_sample_point(p)

    points_being_sampled = [-1.0, 1.0]

    print "about to step"

    for _ in range(number_of_steps):
        # one returns!
        if 1.0 in points_being_sampled:
            returned_point = 1
        elif -1.0 in points_being_sampled:
            returned_point = 0
        else:
            returned_point = numpy.random.randint(0,2) # choose either 0 or 1
        point = points_being_sampled[returned_point]
        points_being_sampled.pop(returned_point)
        p = SamplePoint(point, true_function(point))
        GPP.add_sample_point(p)

        multistart_points = numpy.random.uniform(low = numpy.min(domain), high = numpy.max(domain), size = 20)
        best_point, best_improvement, polyak_ruppert_paths = GPP.get_next_step_multistart(multistart_points, points_being_sampled, domain)
        best_point_fmin = GPP.get_fmin_multistart(multistart_points, points_being_sampled, domain)

        print "best point: %f" % best_point

        fig = plt.figure(figsize=(12,5))

        fig = GPP.plot(domain=numpy.arange(-1.0,1.01,0.01), points_being_sampled=points_being_sampled, true_function=true_function, save_figure=True, passed_fig=fig, fig_place=321)
        fig = GPP.plot_contour_and_quiver(domain=numpy.arange(-1.0,1.05,0.05), points_being_sampled=points_being_sampled, next_point_to_be_sampled=best_point, save_figure=True, passed_fig=fig, fig_place=122)
        fig = GPP.plot_1D_EI_during_sample(domain=numpy.arange(-1.0,1.01,0.01), points_being_sampled=points_being_sampled, next_points_to_be_sampled=[best_point, best_point_fmin], save_figure=True, passed_fig=fig, fig_place=323)

        ax = fig.add_subplot(325)
        for path in polyak_ruppert_paths:
            ax.plot(range(len(path[0])), path[0])
        ax.plot([numpy.min(domain), numpy.max(domain)], [best_point, best_point])

        pdf_stream.savefig()

        print "making next step"



        points_being_sampled.append(best_point)



    pdf_stream.close()

def stepper_test():
    def true_function(x):
        return 4*(x - 0.45)**2

    points_to_sample = [-1,-0.5,0,0.5,1]

    covariance_of_process = CovarianceOfProcess(l=0.1, alpha=10.0)

    GPP = GaussianProcess(covariance_of_process=covariance_of_process)

    pdf_stream = PdfPages("stepper_test.pdf")

    for point in points_to_sample:
        p = SamplePoint(point, true_function(point))
        GPP.add_sample_point(p)

    if 0:
        fig = plt.figure(figsize=(12,5))
        fig = GPP.plot(domain=numpy.arange(-1.0,1.01,0.01), true_function=true_function, save_figure=True, passed_fig=fig)
        fig = GPP.plot_contour_and_quiver(domain=numpy.arange(-1.0,1.05,0.05), save_figure=True, passed_fig=fig)
        fig = GPP.plot_1D_EI(domain=numpy.arange(-1.0,1.01,0.01), save_figure=True, passed_fig=fig)
        pdf_stream.savefig()

        print "Sampled point %f" % point

    print GPP.get_next_step(0.1, [0.75])

    print GPP.get_next_step_fmin(0.1, [0.75])

    pdf_stream.close()

def figure_three_GiLeCa08_test():
    def true_function(x):
        return 4*(x - 0.45)**2

    #points_to_sample = [numpy.array([-1]),numpy.array([-0.5]),numpy.array([0]),numpy.array([0.5]),numpy.array([1])]
    points_to_sample = [numpy.array([0]),numpy.array([-1]),numpy.array([1]),numpy.array([-0.5]),numpy.array([0.5])]

    covariance_of_process = CovarianceOfProcess(l=0.1, alpha=10.0)

    GPP = GaussianProcess(covariance_of_process=covariance_of_process)

    pdf_stream = PdfPages("GiLeCa08_fig3.pdf")

    for point in points_to_sample:
        p = SamplePoint(point, true_function(point))
        GPP.add_sample_point(p)

        fig = plt.figure(figsize=(12,5))
        fig = GPP.plot(domain=numpy.arange(-1.0,1.01,0.01), true_function=true_function, save_figure=True, passed_fig=fig)
        fig = GPP.plot_contour_and_quiver(domain=numpy.arange(-1.0,1.05,0.05), save_figure=True, passed_fig=fig)
        fig = GPP.plot_1D_EI(domain=numpy.arange(-1.0,1.01,0.01), save_figure=True, passed_fig=fig)
        pdf_stream.savefig()

        print "Sampled point %f" % point

    pdf_stream.close()


def simple_test(points_to_sample=None, true_function=None, make_gif=False, figure_name="GPP_evolution.pdf"):
    """Given a function and points to sample this will plot the GPP as the points are added"""
    # use default func if needed
    def sample_function(x):
        """A meandering sine wave"""
        return -0.1*(numpy.sin(x*(2*numpy.pi)) + 2*numpy.sin(x*1.5)) + .3
    def sample_function_two(x):
        if x < 0.75:
            return -x + 0.5
        elif x < 1.25:
            return x - 1
        elif x < 1.75:
            return -x + 1.5
        else:
            return x - 2
    if not true_function:
        true_function = sample_function

    # use default sample points if needed
    if not points_to_sample:
        points_to_sample = [1.0, 2.5, 0.0, 3.0, 2.0, 1.5, 0.5]

    GPP = GaussianProcess()

    if make_gif:
        pdf_stream = PdfPages(figure_name)

    # evolve the GPP
    for point in points_to_sample:
        p = SamplePoint(point, true_function(point))
        GPP.add_sample_point(p)

        if not make_gif:
            GPP.plot(true_function=true_function)
            GPP.plot_contour_and_quiver(domain=numpy.arange(0.0,3.1,0.1))
        else:
            fig = plt.figure()
            fig = GPP.plot(true_function=true_function, save_figure=True, passed_fig=fig)
            fig = GPP.plot_contour_and_quiver(domain=numpy.arange(0.0,3.1,0.1), save_figure=True, passed_fig=fig)
            pdf_stream.savefig()


        print "Finished point %f" % point

    if make_gif:
        pdf_stream.close()


def get_default_GPP(points_to_sample=None, true_function=None):
    """Given a function and points to sample this will plot the GPP as the points are added"""
    # use default func if needed
    def sample_function(x):
        return -0.1*(numpy.sin(x*(2*numpy.pi)) + 2*numpy.sin(x*1.5)) + .3
    if not true_function:
        true_function = sample_function

    # use default sample points if needed
    if not points_to_sample:
        points_to_sample = [1.0, 2.5, 0.0, 3.0, 2.0, 1.5, 0.5]

    GPP = GaussianProcess()
    # evolve the GPP
    for point in points_to_sample:
        p = SamplePoint(point, true_function(point))
        GPP.add_sample_point(p)

    return GPP

def cholesky_decomp_test():
    GPP = get_default_GPP()

    for x1 in [0.4,0.6,0.7]:
        for x2 in [0.4,0.6,0.7]:
            if x1 != x2:
                mu, var = GPP.get_mean_and_var_of_points([x1, x2])
                #grad_mu = GPP.get_grad_mu([x1, x2])
                cholesky_decomp, grad_chol_dec_one = GPP.get_cholesky_decomp([x1, x2], 0)
                cholesky_decomp, grad_chol_dec_two = GPP.get_cholesky_decomp([x1, x2], 1)
                EI = GPP.get_expected_improvement([x1,x2], iterations=1000)
                print ""
                print "Sampling points (%f,%f)" % (x1, x2)
                print "mu1 = %f, mu2 = %f, var00 = %f, var11 = %f" % (mu[0], mu[1], var[0][0], var[1][1])
                print "EI = %f" % EI
                print "Cholesky Decomp:"
                print cholesky_decomp
                print "grad_cholesky_decomp wrt x1"
                print grad_chol_dec_one
                print "grad_cholesky_decomp wrt x2"
                print grad_chol_dec_two

def quiver_test(points_to_sample=None, true_function=None):
    """Given a function and points to sample this will plot the GPP as the points are added"""
    # use default func if needed
    def sample_function(x):
        return -0.1*(numpy.sin(x*(2*numpy.pi)) + 2*numpy.sin(x*1.5)) + .3
    if not true_function:
        true_function = sample_function

    # use default sample points if needed
    if not points_to_sample:
        points_to_sample = [1.0, 2.5, 0.0, 3.0, 2.0, 1.5, 0.5]

    GPP = GaussianProcess()
    # evolve the GPP
    for point in points_to_sample:
        p = SamplePoint(point, true_function(point))
        GPP.add_sample_point(p)

    GPP.plot(true_function=true_function)
    GPP.plot_contour_and_quiver(domain=numpy.arange(0.0,3.05,0.05))

def branin_fmin_test(number_of_attempts=100):
    def branin_func(input_vec):
        x1 = input_vec[0]
        x2 = input_vec[1]
        # enforce boundary conditions
        if x1 > 10 or x1 < -5 or x2 < 0 or x2 > 15:
            return (x2 - (5.1/(4*numpy.pi**2))*x1**2 + (5/numpy.pi)*x1 - 6)**2 + 10*(1 - 1.0/(8*numpy.pi))*numpy.cos(x1) + 10 + x1**4 + x2**4
        return (x2 - (5.1/(4*numpy.pi**2))*x1**2 + (5/numpy.pi)*x1 - 6)**2 + 10*(1 - 1.0/(8*numpy.pi))*numpy.cos(x1) + 10

    def make_rand_point(domain_x1, domain_x2):
        return numpy.array([numpy.random.uniform(low = numpy.min(domain_x1), high = numpy.max(domain_x1)), numpy.random.uniform(low = numpy.min(domain_x2), high = numpy.max(domain_x2))])

    evals = []

    for _ in range(number_of_attempts):
        evals.append(fmin(branin_func, make_rand_point([-5,10], [0,15]), xtol=1e-4, full_output = 1)[3])

    print evals
    print numpy.min(evals)
    print numpy.max(evals)
    print numpy.std(evals)
    print numpy.median(evals)
    print numpy.mean(evals)

def branin_func(input_vec):
    # http://www.it.lut.fi/ip/evo/functions/node27.html
    # min : 0.397887346
    # at
    # x1 = -3.141592, x2 = 12.274999
    # x1 = 3.141592, x2 = 2.275000
    # x1 = 9.424777, x2 = 2.474999
    x1 = input_vec[0]
    x2 = input_vec[1]
    #return (x1-3)**2 + (x2-2)**2
    # enforce boundary conditions
    if x1 > 10 or x1 < -5 or x2 < 0 or x2 > 15:
        raise(ValueError, "Outside of domain of branin function x1 in [-5,10] and x2 in [0,15]")
        #return (x2 - (5.1/(4*numpy.pi**2))*x1**2 + (5/numpy.pi)*x1 - 6)**2 + 10*(1 - 1.0/(8*numpy.pi))*numpy.cos(x1) + 10 + x1**4 + x2**4
    return (x2 - (5.1/(4*numpy.pi**2))*x1**2 + (5/numpy.pi)*x1 - 6)**2 + 10*(1 - 1.0/(8*numpy.pi))*numpy.cos(x1) + 10

def overnight_setup():

    covariance_of_process = CovarianceOfProcess(l=2.0, alpha=10.0)

    GPP = GaussianProcess(covariance_of_process=covariance_of_process)

    sample_edges = True
    if sample_edges:
        for x1 in [-5, 2.5, 10]:
            for x2 in [0, 7.5, 15]:
                p = SamplePoint(numpy.array([x1, x2]), branin_func([x1, x2]))
                #print p
                GPP.add_sample_point(p)
    else:
        # bootstrap sample middle
        p = SamplePoint(numpy.array([2.5, 7.5]), branin_func([2.5, 7.5]))
        #print p
        GPP.add_sample_point(p)

    return GPP



def overnight_test(processors=1, number_of_samples=24):
    print "Starting test p=%i= s=%i= ring tested" % (processors, number_of_samples)
    GPP = overnight_setup()

    points_sampling = []
    points_sampling.append(get_multistart_best_fmin(GPP, points_being_sampled=points_sampling))
    for _ in range(processors - 1):
        points_sampling.append(get_multistart_best_fmin(GPP, points_being_sampled=points_sampling))


    for _ in range(number_of_samples):
        t0 = time.time()
        # one comes back
        returned_point = numpy.random.randint(0,processors) # choose either 0 or 1
        point = points_sampling[returned_point]
        points_sampling.pop(returned_point)
        # sample it
        p = SamplePoint(point, branin_func(point))
        print p
        GPP.add_sample_point(p)
        # pick a new point to sample
        points_sampling.append(get_multistart_best_fmin(GPP, points_being_sampled=points_sampling))
        #print "sampling ", point
        print time.time() - t0


def overnight_runner(tests, procs):
    for _ in range(tests):
        overnight_test(processors=procs)


def branin_test(plot_branin=True):
    """Test the algorithm against the Branin Function"""


    print "fmin"
    #print fmin(branin_func, numpy.array([4,4]), xtol=1e-4)

    if plot_branin:
        res = 0.1
        domain_x1 = numpy.arange(-5, 10 + res, res)
        domain_x2 = numpy.arange(0, 15 + res, res)

        branin_contour = numpy.zeros((len(domain_x1), len(domain_x2)))

        for i, point_x1 in enumerate(domain_x1):
            for j, point_x2 in enumerate(domain_x2):
                branin_contour[i][j] = branin_func([point_x1, point_x2])

        X, Y = numpy.meshgrid(domain_x2, domain_x1)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        CS = plt.contour(Y, X, branin_contour)

        ax.plot([-3.141592, 3.141592, 9.424777], [12.274999, 2.275000, 2.474999], 'bx')

    # start the GPP
    covariance_of_process = gp.CovarianceOfProcess(l=2.0, alpha=10.0)

    GPP = gp.GaussianProcess(covariance_of_process=covariance_of_process)

    sample_edges = True
    if sample_edges:
        for x1 in [-5, 2.5, 10]:
            for x2 in [0, 7.5, 15]:
                p = gp.SamplePoint(numpy.array([x1, x2]), branin_func([x1, x2]))
                print p
                GPP.add_sample_point(p)
                if plot_branin:
                    ax.plot([p.point[0]],[p.point[1]],'rx')
    else:
        # bootstrap sample middle
        p = gp.SamplePoint(numpy.array([2.5, 7.5]), branin_func([2.5, 7.5]))
        print p
        GPP.add_sample_point(p)
        if plot_branin:
            ax.plot([p.point[0]],[p.point[1]],'rx')

    print "best_so_far", GPP.best_so_far

    if plot_branin:
        ax.plot([p.point[0]],[p.point[1]],'rx')

    #p = SamplePoint(numpy.array([1,0]), branin_func([1,0]))
    #print p
    #GPP.add_sample_point(p)

    #print GPP.get_expected_improvement([numpy.array([0,1.01])], iterations=10)

    def make_rand_point(domain_x1, domain_x2):
        return numpy.array([numpy.random.uniform(low = numpy.min(domain_x1), high = numpy.max(domain_x1)), numpy.random.uniform(low = numpy.min(domain_x2), high = numpy.max(domain_x2))])

    def get_multistart_best(GPP, random_restarts=5, points_being_sampled=[]):
        best_improvement = -numpy.inf
        #random_restarts = len(GPP.points_sampled)*4
        for j in range(random_restarts):
            path, next_step, improvement = GPP.get_next_step(make_rand_point(domain_x1, domain_x2), points_being_sampled, [0,1])
            #path, next_step, improvement = GPP.get_next_step(numpy.array([-2.141592, 11.274999]), [numpy.array([1.0,0.0])], [0,1])
            if improvement > best_improvement:
                best_improvement = improvement
                best_next_step = next_step
        return best_next_step

    processors = 4
    number_of_samples = 5

    points_sampling = []
    points_sampling.append(get_multistart_best())
    for _ in range(processors - 1):
        points_sampling.append(get_multistart_best(points_being_sampled=points_sampling))

    for point in points_sampling:
        print "sampling ", point

    for i in range(number_of_samples):
        t0 = time.time()
        # one comes back
        returned_point = numpy.random.randint(0,processors) # choose either 0 or 1
        point = points_sampling[returned_point]
        points_sampling.pop(returned_point)
        # sample it
        p = SamplePoint(point, branin_func(point))
        print p
        GPP.add_sample_point(p)
        # pick a new point to sample
        points_sampling.append(get_multistart_best(points_being_sampled=points_sampling))
        print "sampling ", point
        print time.time() - t0

    """
    random_restarts = 5
    number_of_samples = 10

    for i in range(number_of_samples):
        best_improvement = -numpy.inf
        #random_restarts = len(GPP.points_sampled)*4
        for j in range(random_restarts):
            path, next_step, improvement = GPP.get_next_step(make_rand_point([2, 4], [1,3]), [numpy.array([1.0,0.0])], [0,1])
            #path, next_step, improvement = GPP.get_next_step(numpy.array([-2.141592, 11.274999]), [numpy.array([1.0,0.0])], [0,1])
            if improvement > best_improvement:
                best_improvement = improvement
                best_path = path
                best_next_step = next_step
        p = SamplePoint(best_next_step, branin_func(best_next_step))
        print p
        GPP.add_sample_point(p)
        x_vals = []
        y_vals = []
        for point in best_path:
            x_vals.append(point[0])
            y_vals.append(point[1])
        if plot_branin:
            ax.plot(x_vals, y_vals, 'r-')
            ax.plot([p.point[0]],[p.point[1]],'rx')
    """
    if plot_branin:
        ax.set_xlim(numpy.min(domain_x1), numpy.max(domain_x1))
        ax.set_ylim(numpy.min(domain_x2), numpy.max(domain_x2))

def get_random_1D_sine_function(components=10, max_freq=10, plot_func=False):
    """Return a function that is a random composition of sine waves"""
    freqs = numpy.random.uniform(low=-max_freq, high=max_freq, size=components)
    shifts = numpy.random.uniform(low=-numpy.pi, high=numpy.pi, size=components)
    signs = numpy.random.randint(low=0, high=2, size=components)
    signs = 2*signs - 1

    def rand_func(x):
        func_val = 0.0
        for i in range(components):
            func_val += signs[i]*numpy.sin(x*freqs[i] + shifts[i])
        return func_val/(0.5*components)

    if plot_func:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(numpy.arange(-1,1,0.01), rand_func(numpy.arange(-1,1,0.01)))
        plt.show()

    return rand_func

def test_c_func():
    covariance_of_process = gp.CovarianceOfProcess(l = 0.1)

    GPP = gp.GaussianProcess(covariance_of_process=covariance_of_process)

    p = gp.SamplePoint(numpy.array([0.0]), 0.0)
    GPP.add_sample_point(p)
    p = gp.SamplePoint(numpy.array([0.25]), 1.0)
    GPP.add_sample_point(p)
    p = gp.SamplePoint(numpy.array([0.5]), -1.0)
    GPP.add_sample_point(p)
    p = gp.SamplePoint(numpy.array([0.75]), 0.0)
    GPP.add_sample_point(p)

    tmp_points_to_sample = numpy.arange(0.0,1.0,0.01);
    points_to_sample = []
    for p in tmp_points_to_sample:
        points_to_sample.append(numpy.array([p]))
    mus, var = GPP.get_mean_and_var_of_points(points_to_sample)
    for i, mu in enumerate(mus):
        print mu, var[i][i]

    g_mus = GPP.get_grad_mu(points_to_sample)
    L, g_var = GPP.cholesky_decomp_and_grad(points_to_sample)

    print g_var

    for i, g_mu in enumerate(g_mus):
        print g_mu[0], g_var[i][0][0]

    t0 = time.time()
    vals = GPP_updater.get_next_step_multistart(GPP, points_to_sample, [], domain=None, gamma=0.8, iterations=100, max_N=100)
    print vals[0]
    print "time", time.time() - t0

    #plot_GPP(GPP, domain=numpy.arange(0,1,0.01), fig_place=111)



def sample_random_function(true_func=None, samples=10, processors=1, domain=[[-1,1]], random_return=False, plot_func=True, save_figure=False):
    """Sample a random function (or true_func) a [samples] times with [processors] over [domain] and return the improvement achieved over a certain number of steps"""

    t0 = time.time()

    if not true_func:
        true_func = get_random_1D_sine_function()

    if plot_func and save_figure:
        pdf_stream = PdfPages("sample_random_function.pdf")

    covariance_of_process = gp.CovarianceOfProcess(l = 0.1)

    GPP = gp.GaussianProcess(covariance_of_process=covariance_of_process)

    # bootstrap: sample edges
    print "bootstrapping p=%i s=%i" % (processors, samples)
    p = gp.SamplePoint(numpy.array([numpy.min(domain)]), true_func(numpy.min(domain)))
    GPP.add_sample_point(p)
    print p

    if plot_func:
        plot_GPP(GPP, domain=numpy.arange(-1,1,0.01), true_function=true_func, save_figure=True, pdf_stream=pdf_stream, fig_place=111)

    p = gp.SamplePoint(numpy.array([numpy.max(domain)]), true_func(numpy.max(domain)))
    GPP.add_sample_point(p)
    print p

    if plot_func:
        plot_GPP(GPP, domain=numpy.arange(-1,1,0.01), true_function=true_func, save_figure=True, pdf_stream=pdf_stream, fig_place=111)

    print "best so far: %f" % GPP.best_so_far

    # build set of initial samples
    print "Getting initial samples"
    points_sampling = []
    points_sampling.append(GPP_updater.get_multistart_best(GPP, domain, random_restarts=4, points_being_sampled=points_sampling))
    for i in range(processors - 1):
        points_sampling.append(GPP_updater.get_multistart_best(GPP, domain, random_restarts=(i+3)*2, points_being_sampled=points_sampling))

    step_improvement = numpy.zeros(samples)

    for i in range(samples):
        # one comes back
        if random_return:
            returned_point = numpy.random.randint(0,processors)
        else:
            returned_point = 0
        point = points_sampling[returned_point]
        points_sampling.pop(returned_point)

        # sample it
        p = gp.SamplePoint(point, true_func(point))
        print p
        if plot_func:
            plot_GPP(GPP, domain=numpy.arange(-1,1,0.01), true_function=true_func, save_figure=True, pdf_stream=pdf_stream, fig_place=111)
        step_improvement[i:] += numpy.max([0, GPP.best_so_far - p.value])
        GPP.add_sample_point(p)

        # pick a new point to sample
        if i < samples - 1:
            points_sampling.append(GPP_updater.get_multistart_best(GPP, domain, random_restarts=len(GPP.points_sampled)*8, points_being_sampled=points_sampling))

    print "Took (seconds): %f" % float(time.time() - t0)

    if save_figure:
        pdf_stream.close()

    return step_improvement

def sample_test(samples=8):
    """Test sampler with a plot"""

    true_func = get_random_1D_sine_function()

    step_imp_1 = sample_random_function(samples=samples, processors=1, true_func=true_func, plot_func=True)

    #step_imp_2 = sample_random_function(samples=2*samples, processors=2, true_func=true_func)

    step_imp_4 = sample_random_function(samples=4*samples, processors=4, true_func=true_func)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1,samples+1), step_imp_1)
    #ax.plot(range(1,2*samples+1), step_imp_2)
    #ax.plot(range(1,4*samples+1), step_imp_4)

def main():
    #sample_random_function(save_figure=True)
    pass

if __name__ == '__main__':
    main()
