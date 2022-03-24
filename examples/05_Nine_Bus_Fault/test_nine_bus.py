#!python3
#
# Copyright (C) 2014-2015 Julius Susanto. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""
PYPOWER-Dynamics
Nine-Bus Network Stability Test

"""
# Dynamic model classes
from pydyn.interface import init_interfaces
from pydyn.mod_Ybus import mod_Ybus
from pydyn.version import pydyn_ver

from scipy.sparse.linalg import splu
import numpy as np

from pypower.runpf import runpf
from pypower.ext2int import ext2int
from pypower.makeYbus import makeYbus
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, \
    VM, VA, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN, REF
from pydyn.controller import controller
from pydyn.sym_order6a import sym_order6a
from pydyn.sym_order6b import sym_order6b
from pydyn.sym_order4 import sym_order4
from pydyn.ext_grid import ext_grid

# Simulation modules
from pydyn.events import events
from pydyn.recorder import recorder
#from pydyn.run_sim import run_sim

# External modules
from pypower.loadcase import loadcase
import matplotlib.pyplot as plt
import numpy as np

def run_sim(ppc, elements, dynopt = None, events = None, recorder = None):
    """
    Run a time-domain simulation
    
    Inputs:
        ppc         PYPOWER load flow case
        elements    Dictionary of dynamic model objects (machines, controllers, etc) with Object ID as key
        events      Events object
        recorder    Recorder object (empty)
    
    Outputs:
        recorder    Recorder object (with data)
    """
    
    #########
    # SETUP #
    #########
    global h,t_sim,max_err,max_iter,verbose,t,interfaces
    # Get version information
    ver = pydyn_ver()
    print('PYPOWER-Dynamics ' + ver['Version'] + ', ' + ver['Date'])
    
    # Program options
    if dynopt:
        h = dynopt['h']             
        t_sim = dynopt['t_sim']           
        max_err = dynopt['max_err']        
        max_iter = dynopt['max_iter']
        verbose = dynopt['verbose']
    else:
        # Default program options
        h = 0.01                # step length (s)
        t_sim = 5               # simulation time (s)
        max_err = 0.0001        # Maximum error in network iteration (voltage mismatches)
        max_iter = 25           # Maximum number of network iterations
        verbose = False
        
    # Make lists of current injection sources (generators, external grids, etc) and controllers
    sources = []
    controllers = []
    for element in elements.values():
        if element.__module__ in ['pydyn.sym_order6a', 'pydyn.sym_order6b', 'pydyn.sym_order4', 'pydyn.ext_grid', 'pydyn.vsc_average', 'pydyn.asym_1cage', 'pydyn.asym_2cage']:
            sources.append(element)
            
        if element.__module__ == 'pydyn.controller':
            controllers.append(element)
    
    # Set up interfaces
    interfaces = init_interfaces(elements)
    
    
    ##################
    # INITIALISATION #
    ##################
    print('Initialising models...')
    
    # Run power flow and update bus voltages and angles in PYPOWER case object
    results, success = runpf(ppc) 
    ppc["bus"][:, VM] = results["bus"][:, VM]
    ppc["bus"][:, VA] = results["bus"][:, VA]
    
    # Build Ybus matrix
    ppc_int = ext2int(ppc)
    baseMVA, bus, branch = ppc_int["baseMVA"], ppc_int["bus"], ppc_int["branch"]
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    
    # Build modified Ybus matrix
    Ybus = mod_Ybus(Ybus, elements, bus, ppc_int['gen'], baseMVA)
    
    # Calculate initial voltage phasors
    v0 = bus[:, VM] * (np.cos(np.radians(bus[:, VA])) + 1j * np.sin(np.radians(bus[:, VA])))
    # Initialise sources from load flow
    for source in sources:
        if source.__module__ in ['pydyn.asym_1cage', 'pydyn.asym_2cage']:
            # Asynchronous machine
            source_bus = int(ppc_int['bus'][source.bus_no,0])
            v_source = v0[source_bus]
            source.initialise(v_source,0)
        else:
            # Generator or VSC
            source_bus = int(ppc_int['gen'][source.gen_no,0])
            S_source = np.complex(results["gen"][source.gen_no, 1] / baseMVA, results["gen"][source.gen_no, 2] / baseMVA)
            v_source = v0[source_bus]
            source.initialise(v_source,S_source)
    
    # Interface controllers and machines (for initialisation)
    for intf in interfaces:
        int_type = intf[0]
        var_name = intf[1]
        if int_type == 'OUTPUT':
            # If an output, interface in the reverse direction for initialisation
            intf[2].signals[var_name] = intf[3].signals[var_name]
        else:
            # Inputs are interfaced in normal direction during initialisation
            intf[3].signals[var_name] = intf[2].signals[var_name]
    
    # Initialise controllers
    for controller in controllers:
        controller.initialise()
    
    #############
    # MAIN LOOP #
    #############
    
    if events == None:
        print('Warning: no events!')
    
    # Factorise Ybus matrix
    Ybus_inv = splu(Ybus)
    
    y1 = []
    v_prev = v0
    print('Simulating...')
    #print('------------')
    for t in range(int(t_sim / h) + 1):
        if np.mod(t,1/h) == 0:
            print('t=' + str(t*h) + 's')
            
        # Interface controllers and machines
        for intf in interfaces:
            var_name = intf[1]
            intf[3].signals[var_name] = intf[2].signals[var_name]
        
        # Solve differential equations
        for j in range(4):
            # Solve step of differential equations
            for element in elements.values():
                element.solve_step(h,j) 
            
            # Interface with network equations
            v_prev = solve_network(sources, v_prev, Ybus_inv, ppc_int, len(bus), max_err, max_iter)
        
        if recorder != None:
            # Record signals or states
            #print(recorder,t*h,elements)
            recorder.record_variables(t*h, elements)
        
        if events != None:
            # Check event stack
            ppc, refactorise = events.handle_events(np.round(t*h,5), elements, ppc, baseMVA)
            
            if refactorise == True:
                # Rebuild Ybus from new ppc_int
                ppc_int = ext2int(ppc)
                baseMVA, bus, branch = ppc_int["baseMVA"], ppc_int["bus"], ppc_int["branch"]
                Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
                
                # Rebuild modified Ybus
                Ybus = mod_Ybus(Ybus, elements, bus, ppc_int['gen'], baseMVA)
                
                # Refactorise Ybus
                Ybus_inv = splu(Ybus)
                
                # Solve network equations
                v_prev = solve_network(sources, v_prev, Ybus_inv, ppc_int, len(bus), max_err, max_iter)
                
    return recorder


def solve_network(sources, v_prev, Ybus_inv, ppc_int, no_buses, max_err, max_iter):
    """
    Solve network equations
    """
    verr = 1
    i = 1
    # Iterate until network voltages in successive iterations are within tolerance
    while verr > max_err and i < max_iter:        
        # Update current injections for sources
        I = np.zeros(no_buses, dtype='complex')
        for source in sources:
            if source.__module__ in ['pydyn.asym_1cage', 'pydyn.asym_2cage']:
                # Asynchronous machine
                source_bus = int(ppc_int['bus'][source.bus_no,0])
            else:
                # Generators or VSC
                source_bus = int(ppc_int['gen'][source.gen_no,0])
                
            I[source_bus] = source.calc_currents(v_prev[source_bus])
            
        
        # Solve for network voltages
        vtmp = Ybus_inv.solve(I) 
        verr = np.abs(np.dot((vtmp-v_prev),np.transpose(vtmp-v_prev)))
        v_prev = vtmp
        i = i + 1
    
    if i >= max_iter:
        print('Network voltages and current injections did not converge in time step...')
    
    return v_prev
    
if __name__ == '__main__':
    
    #########
    # SETUP #
    #########
    
    print('---------------------------------------')
    print('PYPOWER-Dynamics - 9 Bus Stability Test')
    print('---------------------------------------')

    # Load PYPOWER case
    ppc = loadcase('case9.py')
    
    # Program options
    dynopt = {}
    dynopt['h'] = 0.01                # step length (s)
    dynopt['t_sim'] = 5.0             # simulation time (s)
    dynopt['max_err'] = 1e-6          # Maximum error in network iteration (voltage mismatches)
    dynopt['max_iter'] = 25           # Maximum number of network iterations
    dynopt['verbose'] = False         # option for verbose messages
    dynopt['fn'] = 60                 # Nominal system frequency (Hz)
    dynopt['speed_volt'] = True       # Speed-voltage term option (for current injection calculation)
    
    # Integrator option
    #dynopt['iopt'] = 'mod_euler'
    dynopt['iopt'] = 'runge_kutta'
          
    # Create dynamic model objects
    G1 = sym_order6b('G1.mach', dynopt)
    G2 = sym_order6b('G2.mach', dynopt)
    G3 = sym_order6b('G3.mach', dynopt)
    
    # Create dictionary of elements
    elements = {}
    elements[G1.id] = G1
    elements[G2.id] = G2
    elements[G3.id] = G3
    #elements[oCtrl.id] = oCtrl
    
    # Create event stack
    oEvents = events('events.evnt')
    
    # Create recorder object
    oRecord = recorder('recorder.rcd')
    global h,t_sim,max_err,max_iter,verbose  
    # Run simulation
    oRecord = run_sim(ppc,elements,dynopt,oEvents,oRecord)
    '''
    # Calculate relative rotor angles
    rel_delta1 = np.array(oRecord.results['GEN2:delta']) - np.array(oRecord.results['GEN1:delta'])
    rel_delta2 = np.array(oRecord.results['GEN3:delta']) - np.array(oRecord.results['GEN1:delta']) 
    
    # Plot variables
    plt.plot(oRecord.t_axis,rel_delta1 * 180 / np.pi, 'r-', oRecord.t_axis, rel_delta2 *180 / np.pi, 'b-')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotor Angles (relative to GEN1)')
    plt.show()
    
    # Write recorded variables to output file
    oRecord.write_output('output.csv')
    '''