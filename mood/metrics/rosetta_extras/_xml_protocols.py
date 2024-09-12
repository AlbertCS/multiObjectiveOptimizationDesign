from mood.metrics.rosetta_extras import rosettaScripts as rs


class xmlProtocols:

    def relaxAroundLigand(ligand_chain, protein_chains, score_function='ref2015',
                          relax_cycles=5, neighbour_distance=12.0, distances=None,
                          interfaces=None, calculate_apo_energy=True, native_state=None,
                          residues_energy=None, cst_files=None, ligand_rmsd=False,
                          null=False):
        """
        Creates an XML protocol that relax the sorrounding residues around an specified
        ligand residue. A list of distances can be defined to be reported at the score
        file.

        Parameters
        ==========
        target_residue : int
            Rosetta index of the ligand residue
        score_function : str
            Name of the score funtion to use.
        relax_cycles : int
            Number of relax cycles to apply.
        neighbour_distance : float
            The distance to the ligand for considering neighbours
        distances : list
            A list of nested tuples each tuple contain two 3-element (residue, chain, index)
            tuples for defining atomic distances to report.
        cst_files : (list, str)
            Give a list or a single constraint file to be applied during the optimisation protocol.
        interfaces : list
            A list chains to calculate their interface scores.
        calculate_apo_energy : bool
            Add protocol for calculating apo energies.
        native_state : int
            Define a state to serve as native structure for RMSD calculation.
        residues_energy : dict
            A dictionary containing tuples defining energy-by-residues calculations. The first tuple
            element is the lists of residue indexes for which to compute their added scores, and the second
            element is the scoretype. If instead of a tuple only a list is given, then the score type
            is assumed as the total_score. The labels of the dictionary is used as the label for each
            computed metric.
        ligand_rmsd : bool
            Compute ligand RMSD
        null : str
            Run the protocol only with a null mover (debug)
        """

        xml_scripts = {}

        # Initialise XML script
        xml_scripts['main'] = rs.xmlScript()
        protocol = []

        # Create all-atom score function
        sfxn = rs.scorefunctions.new_scorefunction(score_function,
                                                   weights_file=score_function)

        # Add cst files to the protocol
        set_cst = {}
        if cst_files != None:
            if isinstance(cst_files, str):
                cst_files = [cst_files]
            set_cst = _addConstraintFiles(xml_scripts['main'], cst_files, sfxn)

            for cst in set_cst:
                protocol.append(set_cst[cst])

        xml_scripts['main'].addScorefunction(sfxn)

        # Create ligand selector
        ligand_selector = rs.residueSelectors.chainSelector('ligand', ligand_chain)
        xml_scripts['main'].addResidueSelector(ligand_selector)

        # Create neighbour selector
        ligand_neighbours = rs.residueSelectors.neighborhood('ligand_neighbours', 'ligand',
                                                             distance=neighbour_distance)
        xml_scripts['main'].addResidueSelector(ligand_neighbours)

        ligand_and_neighbours = rs.residueSelectors.orSelector('ligand_and_neighbours',
                                                               selectors=['ligand', 'ligand_neighbours'])
        xml_scripts['main'].addResidueSelector(ligand_and_neighbours)

        not_ligand_or_neighbours = rs.residueSelectors.notSelector('not_ligand_or_neighbours', 'ligand_and_neighbours')
        xml_scripts['main'].addResidueSelector(not_ligand_or_neighbours)

        # Create movemap factory
        mmf = rs.moveMapFactory('relax_mm')
        mmf.addBackboneOperation(enable=False, residue_selector=not_ligand_or_neighbours)
        mmf.addBackboneOperation(enable=True, residue_selector=ligand_neighbours)
        # mmf.addChiOperation(enable=False, residue_selector=not_ligand_or_neighbours)
        # mmf.addChiOperation(enable=True, residue_selector=ligand_neighbours)
        xml_scripts['main'].addMoveMapFactory(mmf)

        # Create task operations
        tos = []
        prevent_repack_to = rs.taskOperations.operateOnResidueSubset('prevent_repack',
                                                                     'not_ligand_or_neighbours',
                                                                     operation='PreventRepackingRLT')
        xml_scripts['main'].addTaskOperation(prevent_repack_to)
        tos.append(prevent_repack_to)

        # Create relax mover
        relax = rs.movers.fastRelax(repeats=relax_cycles, scorefxn=sfxn, movemap_factory=mmf, task_operations=tos)
        xml_scripts['main'].addMover(relax)
        if not null:
            protocol.append(relax)

        if native_state:
            _calculateNativeRMSD(xml_scripts['main'], protocol, protein_chains)

        # Create interface score calculator mover
        chains = [ligand_chain]
        if interfaces:
            for chain in interfaces:
                if chain != ligand_chain:
                    chains.append(chain)
        _calculateInterface(xml_scripts['main'], protocol, chains, score_function=sfxn)

        if distances:
            if not isinstance(distances, dict):
                raise ValueError('the distances parameter must be a dictionary. Please check the documentation.')
            _calculateDistances(xml_scripts['main'], protocol, distances)

        if residues_energy:
            if not isinstance(residues_energy, dict):
                raise ValueError('residues_energy must be dictionary. Better check the documentation!')
            for label in residues_energy:

                if isinstance(residues_energy[label], tuple) and len(residues_energy[label]) == 2:
                    _calculateResiduesEnergy(xml_scripts['main'], protocol,
                                             residues_energy[label][0], label, score_type=residues_energy[label][1])
                elif isinstance(residues_energy[label], list):
                    _calculateResiduesEnergy(xml_scripts['main'], protocol,
                                             residues_energy[label], label)
                else:
                    raise ValueError('Incorrect residues_energy dictionary entries. Better check the documentation!')

        if ligand_rmsd:
            _calculateLigandRMSD(xml_scripts['main'], protocol, ligand_chain)

        # Set protocol
        xml_scripts['main'].setProtocol(protocol)

        # Add scorefunction output
        xml_scripts['main'].addOutputScorefunction(sfxn)

        # Calculate APO energy
        if calculate_apo_energy:
            if protein_chains == 'None':
                raise ValueError('protein_chains must be given to calculate protein apo energies!')
            xml_scripts['apo'] = _getXMLApoProtocol(protein_chains, score_function=score_function)

        return xml_scripts

    def relaxAroundResidues(target_residues, protein_chains, ligand_chain=None, score_function='ref2015',
                            relax_cycles=5, neighbour_distance=12.0, distances=None,
                            interfaces=None, calculate_apo_energy=True, native_state=None,
                            residues_energy=None, cst_files=None, ligand_rmsd=False, null=False):
        """
        Creates an XML protocol that relax the sorrounding residues around an specified
        ligand residue. A list of distances can be defined to be reported at the score
        file.

        Parameters
        ==========
        target_residue : int
            Rosetta index of the ligand residue
        protein_chains : int
            PDB chain ids of the chains to be included in the protein energy calculations.
        ligand_chain : int
            PDB chain id for the ligand
        score_function : str
            Name of the score funtion to use.
        relax_cycles : int
            Number of relax cycles to apply.
        neighbour_distance : float
            The distance to the ligand for considering neighbours
        distances : list
            A list of nested tuples each tuple contain two 3-element (residue, chain, index)
            tuples for defining atomic distances to report.
        cst_files : (list, str)
            Give a list or a single constraint file to be applied during the optimisation protocol.
        interfaces : list
            A list chains to calculate their interface scores.
        calculate_apo_energy : bool
            Add protocol for calculating apo energies.
        native_state : int
            Define a state to serve as native structure for RMSD calculation.
        residues_energy : dict
            A dictionary containing tuples defining energy-by-residues calculations. The first tuple
            element is the lists of residue indexes for which to compute their added scores, and the second
            element is the scoretype. If instead of a tuple only a list is given, then the score type
            is assumed as the total_score. The labels of the dictionary is used as the label for each
            computed metric.
        ligand_rmsd : bool
            Compute ligand RMSD
        null : str
            Run the protocol only with a null mover (debug)
        """

        xml_scripts = {}

        # Initialise XML script
        xml_scripts['main'] = rs.xmlScript()
        protocol = []

        # Create all-atom score function
        sfxn = rs.scorefunctions.new_scorefunction(score_function,
                                                   weights_file=score_function)

        # Add cst files to the protocol
        set_cst = {}
        if cst_files != None:
            if isinstance(cst_files, str):
                cst_files = [cst_files]
            set_cst = _addConstraintFiles(xml_scripts['main'], cst_files, sfxn)

            for cst in set_cst:
                protocol.append(set_cst[cst])

        xml_scripts['main'].addScorefunction(sfxn)

        # Create ligand selector
        target_residues_selector = rs.residueSelectors.index('target_residues', target_residues)
        xml_scripts['main'].addResidueSelector(target_residues_selector)

        # Create neighbour selector
        target_residues_neighbours = rs.residueSelectors.neighborhood('target_residues_neighbours', 'target_residues',
                                                                      distance=neighbour_distance)
        xml_scripts['main'].addResidueSelector(target_residues_neighbours)

        target_residues_and_neighbours = rs.residueSelectors.orSelector('target_residues_and_neighbours',
                                                                        selectors=['target_residues', 'target_residues_neighbours'])
        xml_scripts['main'].addResidueSelector(target_residues_and_neighbours)

        not_target_residues_or_neighbours = rs.residueSelectors.notSelector('not_target_residues_or_neighbours', 'target_residues_and_neighbours')
        xml_scripts['main'].addResidueSelector(not_target_residues_or_neighbours)

        # Create movemap factory
        mmf = rs.moveMapFactory('relax_mm')
        mmf.addBackboneOperation(enable=False, residue_selector=not_target_residues_or_neighbours)
        mmf.addBackboneOperation(enable=True, residue_selector=target_residues_neighbours)
        # mmf.addChiOperation(enable=False, residue_selector=not_ligand_or_neighbours)
        # mmf.addChiOperation(enable=True, residue_selector=ligand_neighbours)
        xml_scripts['main'].addMoveMapFactory(mmf)

        # Create task operations
        tos = []
        prevent_repack_to = rs.taskOperations.operateOnResidueSubset('prevent_repack',
                                                                     'not_target_residues_or_neighbours',
                                                                     operation='PreventRepackingRLT')
        xml_scripts['main'].addTaskOperation(prevent_repack_to)
        tos.append(prevent_repack_to)

        # Create relax mover
        relax = rs.movers.fastRelax(repeats=relax_cycles, scorefxn=sfxn, movemap_factory=mmf, task_operations=tos)
        xml_scripts['main'].addMover(relax)
        if not null:
            protocol.append(relax)

        if native_state:
            _calculateNativeRMSD(xml_scripts['main'], protocol, protein_chains)

        # Create interface score calculator mover
        if ligand_chain:
            chains = [ligand_chain]
            if interfaces:
                for chain in interfaces:
                    if chain != ligand_chain:
                        chains.append(chain)
            _calculateInterface(xml_scripts['main'], protocol, chains, score_function=sfxn)

        if distances:
            if not isinstance(distances, dict):
                raise ValueError('the distances parameter must be a dictionary. Please check the documentation.')
            _calculateDistances(xml_scripts['main'], protocol, distances)

        if residues_energy:
            if not isinstance(residues_energy, dict):
                raise ValueError('residues_energy must be dictionary. Better check the documentation!')
            for label in residues_energy:

                if isinstance(residues_energy[label], tuple) and len(residues_energy[label]) == 2:
                    _calculateResiduesEnergy(xml_scripts['main'], protocol,
                                             residues_energy[label][0], label, score_type=residues_energy[label][1])
                elif isinstance(residues_energy[label], list):
                    _calculateResiduesEnergy(xml_scripts['main'], protocol,
                                             residues_energy[label], label)
                else:
                    raise ValueError('Incorrect residues_energy dictionary entries. Better check the documentation!')

        if ligand_rmsd:
            if ligand_chain == None:
                raise ValueError('You must set the ligand chain to compute the ligand RMSD')

            _calculateLigandRMSD(xml_scripts['main'], protocol, ligand_chain)

        # Set protocol
        xml_scripts['main'].setProtocol(protocol)

        # Add scorefunction output
        xml_scripts['main'].addOutputScorefunction(sfxn)

        # Calculate APO energy
        if calculate_apo_energy:
            if protein_chains == 'None':
                raise ValueError('protein_chains must be given to calculate protein apo energies!')
            xml_scripts['apo'] = _getXMLApoProtocol(protein_chains, score_function=score_function)

        return xml_scripts

    def fullRelax(protein_chains, score_function='ref2015', relax_cycles=5, distances=None,
                  ligand_chain=None, interfaces=None, null=False, calculate_apo_energy=True,
                  native_state=None, fixed_chains=None, cst_files=None, residues_energy=None,
                  ligand_rmsd=False):
        """
        Creates an XML protocol that relax the full protein system. A list of distances
        can be optionally defined to be reported at the score file. A ligand_chain
        can be given to calculate the interface score after the relaxation has been applied.

        Parameters
        ==========
        score_function : str
            Name of the score funtion to use.
        relax_cycles : int
            Number of relax cycles to apply.
        distances : dict
            A dictionary containing lists of tuple-distances (residue, chain, index)
            as values and labels for keys. All distances in the dictionary are computed.
            The same dictionary can be given to the geneticAlgorithm() function
            which will combined them to used them as filters (see the specfic documentation therin).
        ligand_chain : str
            Ligand chain to calculate its interface score.
        protein_chains : str
            The chains to consider as protein when calcualting the interface score between
            protein and ligand and when considering the Apo energy.
        other_chains : str
            Others chains to be removed when calculating the Apo energies.
        null : str
            Run the protocol only with a null mover (debug)
        calculate_apo_energy : bool
            Add protocol for calculating apo energies.
        native_state : int
            Define a state to serve as native structure for RMSD calculation.
        fixed_chains : list
            Chains to fix during relax.
        cst_files : (list, str)
            Give a list or a single constraint file to be applied during the optimisation protocol.
        residues_energy : dict
            A dictionary containing tuples defining energy-by-residues calculations. The first tuple
            element is the lists of residue indexes for which to compute their added scores, and the second
            element is the scoretype. If instead of a tuple only a list of residues is given, then the score type
            is assumed as the total_score. The labels of the dictionary is used as the label for each
            computed metric.
        ligand_rmsd : bool
            Compute ligand RMSD
        """

        if ligand_chain != None and protein_chains == None:
            message = 'protein_chains must be provided if using ligand_chain interface calculation.'
            raise ValueError(message)

        xml_scripts = {}

        # Initialise XML script
        xml_scripts['main'] = rs.xmlScript()
        protocol = []

        # Create all-atom score function
        sfxn = rs.scorefunctions.new_scorefunction(score_function,
                                                   weights_file=score_function)

        # Add cst files to the protocol
        set_cst = {}
        if cst_files != None:
            if isinstance(cst_files, str):
                cst_files = [cst_files]
            set_cst = _addConstraintFiles(xml_scripts['main'], cst_files, sfxn)

            for cst in set_cst:
                protocol.append(set_cst[cst])

        xml_scripts['main'].addScorefunction(sfxn)

        # Create selectors if needed
        if fixed_chains:
            fixed_chains_selector = rs.residueSelectors.chainSelector('fixed_chains', fixed_chains)
            xml_scripts['main'].addResidueSelector(fixed_chains_selector)

        # Create movemap factory if needed
        if fixed_chains:
            mmf = rs.moveMapFactory('relax_mm')
            mmf.addBackboneOperation(enable=False, residue_selector=fixed_chains_selector)
            xml_scripts['main'].addMoveMapFactory(mmf)

        # Create task operations if needed
        if fixed_chains:
            tos = []
            prevent_repack_to = rs.taskOperations.operateOnResidueSubset('prevent_repack',
                                                                         'fixed_chains',
                                                                         operation='PreventRepackingRLT')
            xml_scripts['main'].addTaskOperation(prevent_repack_to)
            tos.append(prevent_repack_to)

        # Create relax mover

        if fixed_chains:
            relax = rs.movers.fastRelax(repeats=relax_cycles, scorefxn=sfxn, movemap_factory=mmf, task_operations=tos)
        else:
            relax = rs.movers.fastRelax(repeats=relax_cycles, scorefxn=sfxn)

        xml_scripts['main'].addMover(relax)
        if not null:
            protocol.append(relax)

        # Create interface score calculator mover
        chains = []
        if ligand_chain:
            chains.append(ligand_chain)
        if interfaces:
            for chain in interfaces:
                if chain != ligand_chain:
                    chains.append(chain)

        if native_state:
            _calculateNativeRMSD(xml_scripts['main'], protocol, protein_chains)

        if chains != []:
            _calculateInterface(xml_scripts['main'], protocol, chains, score_function=sfxn)

        # Create atomic distances filters with zero confidence for distance evaluations
        if distances:
            _calculateDistances(xml_scripts['main'], protocol, distances)

        if residues_energy:

            if not isinstance(residues_energy, dict):
                raise ValueError('residues_energy must be dictionary. Better check the documentation!')
            for label in residues_energy:

                if isinstance(residues_energy[label], tuple) and len(residues_energy[label]) == 2:
                    _calculateResiduesEnergy(xml_scripts['main'], protocol,
                                             residues_energy[label][0], label, score_type=residues_energy[label][1])
                elif isinstance(residues_energy[label], list):
                    _calculateResiduesEnergy(xml_scripts['main'], protocol,
                                             residues_energy[label], label)
                else:
                    raise ValueError('Incorrect residues_energy dictionary entries. Better check the documentation!')

        if ligand_rmsd:
            if ligand_chain == None:
                raise ValueError('You must set the ligand chain to compute the ligand RMSD')

            _calculateLigandRMSD(xml_scripts['main'], protocol, ligand_chain)

        # Set protocol
        xml_scripts['main'].setProtocol(protocol)

        # Add scorefunction output
        xml_scripts['main'].addOutputScorefunction(sfxn)

        # Calculate APO energy
        if calculate_apo_energy:
            if protein_chains == 'None':
                raise ValueError('protein_chains must be given to calculate protein apo energies!')
            xml_scripts['apo'] = _getXMLApoProtocol(protein_chains, score_function=score_function,
                                                    cst_files=cst_files)

        return xml_scripts

def _calculateNativeRMSD(xml_object, protocol, chains):
    """
    Create a filter for calculating RMSD to a native state structure.
    """

    native_rmsd = rs.filters.rmsd(name='native_rmsd', superimpose_on_all=True, confidence=0.0, chains=chains)

    # xml_object.addMover(read_native)
    xml_object.addFilter(native_rmsd)
    protocol.append(native_rmsd)

def _calculateDistances(xml_object, protocol, distances):
    """
    Create atomic distances filters with zero confidence for distance evaluations.
    """

    atom_distances = {}
    for metric_label in distances:
        for d in distances[metric_label]:
            chain1, resid1, atomname1 = d[0]
            chain2, resid2, atomname2 = d[1]
            label1 = chain1+'_'+str(resid1)+'_'+atomname1
            label2 = chain2+'_'+str(resid2)+'_'+atomname2
            label = 'distance_'+label1+'-'+label2
            atom_distances[label] = rs.filters.atomicDistance(name=label,
                                    residue1=str(resid1)+chain1,
                                    residue2=str(resid2)+chain2,
                                    atomname1=atomname1, atomname2=atomname2,
                                    confidence=0)
            xml_object.addFilter(atom_distances[label])
            protocol.append(atom_distances[label])

def _calculateInterface(xml_object, protocol, chains, score_function='ref2015'):
    """
    Defines an interface score calculator for the given chains.
    """
    interface = rs.movers.interfaceScoreCalculator(chains=chains, scorefxn=score_function)
    xml_object.addMover(interface)
    protocol.append(interface)

def _calculateLigandRMSD(xml_object, protocol, ligand_chain, residue_selector=None):
    """
    Defines an interface score calculator for the given chains.
    """

    if not residue_selector:
        residue_selector = rs.residueSelectors.chainSelector('ligand_selector', ligand_chain)

    ligand_rmsd = rs.simpleMetrics.RMSDMetric(label='ligand_rmsd', residue_selector=residue_selector,
                                              use_native=True)
    xml_object.addResidueSelector(residue_selector)
    xml_object.addSimpleMetric(ligand_rmsd)
    protocol.append(ligand_rmsd)

def _calculateResiduesEnergy(xml_object, protocol, residues, metric_label, score_type='total_score'):
    """
    Compute the energy for the given residues and store in the output scorefile
    as the given score name.

    Parameters
    ==========
    xml_object : rosettaScrtits.xmlScript
        The XML object in which to add the scoring metric
    residues : list
        The list of residues to which calculate their score
    score_type : string
        The score type to calculate for the given residues.
    """

    residue_selector = rs.residueSelectors.index('selector_'+metric_label, residues)
    xml_object.addResidueSelector(residue_selector)

    scoretype_name = ''.join([x.capitalize() for x in score_type.split('_')])

    # Generate metric name of not given
    residues_energy = rs.simpleMetrics.totalEnergyMetric(name=metric_label, label=metric_label,
                                                         residue_selector=residue_selector, scoretype=score_type)

    xml_object.addSimpleMetric(residues_energy)
    protocol.append(residues_energy)

def _addConstraintFiles(xml_object, cst_files, sfxn):
    """
    Add constraint files. The score function must already be added to the xml_object.
    """

    # Create dict for cst_type and cst_term
    # This is a minimal support, it should be expanded as needed.
    cst_types = {
        'AtomPair' : 'atom_pair_constraint',
        'Angle' : 'angle_constraint',
        'Dihedral' : 'dihedral_constraint',
        'CoordinateConstraint' : 'coordinate_constraint'}

    # Gather all terms to be activated in the sfxn
    cst_terms = set()
    set_cst = {}
    zf = len(str(len(cst_files)))
    for i,cst_file in enumerate(cst_files):
        with open(cst_file) as cf:
            for l in cf:
                cst_terms.add(l.split()[0])

        # Add cst_files to xml
        set_cst[cst_file] = rs.movers.constraintSetMover(name='constraintSetMover'+str(i+1).zfill(zf), add_constraints=True,
                                                         cst_file='../../cst_files/'+cst_file.split('/')[-1])

        xml_object.addMover(set_cst[cst_file])

    # Add cst terms to sfxn
    for cst_term in cst_terms:
        sfxn.addReweight(cst_types[cst_term], 1.0)

    return set_cst

def _getXMLApoProtocol(protein_chains, score_function='ref2015', cst_files=None):

    # Initialise XML script
    xml = rs.xmlScript()
    protocol = []

    # Create all-atom score function
    sfxn = rs.scorefunctions.new_scorefunction(score_function,
                                               weights_file=score_function)

    # Add cst files to the apo protocol
    set_cst = {}
    if cst_files != None:
        if isinstance(cst_files, str):
            cst_files = [cst_files]
        set_cst = _addConstraintFiles(xml, cst_files, sfxn)
        for cst in set_cst:
            protocol.append(set_cst[cst])

    xml.addScorefunction(sfxn)

    # Add residue selector for the protein chain
    protein_selector = rs.residueSelectors.chainSelector('protein', protein_chains)
    xml.addResidueSelector(protein_selector)

    # Add residue selector for all not-protein regions
    not_protein_selector = rs.residueSelectors.notSelector('not_protein', 'protein')
    xml.addResidueSelector(not_protein_selector)

    # Delete all not-protein regions
    delete_not_protein = rs.movers.deleteRegionMover(name='delete_not_protein', residue_selector=not_protein_selector)
    xml.addMover(delete_not_protein)
    protocol.append(delete_not_protein)

    # Set protocol
    xml.setProtocol(protocol)

    # Add scorefunction output
    xml.addOutputScorefunction(sfxn)

    return xml
