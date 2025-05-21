classdef AnalyzeTrackingData_withDirection_master<handle

    % Version 2025_05_06 (Last modified by Yuze)
    % - Fix the bug where there is only one fit but tyring to plot 2nd
    % - Change the working unit from nm into µm (e.g. Lc calculation)



    properties

        data             % tracked raw data (in nm)
        datasetIdx       % dataset number (in case of multiple files)
        numTraj          % total number of individual trajectories
        allTraj          % all trajectory data structure
        unconfined       % long detached trajectories
        confined         % confined trajectories
        associationTime  % statistics of association times
        dissociationTime % statistics of dissociation time
        classifier       % the resulting classifier
        displacement = struct('confined',[],...
            'unconfined',[]);
        % Parameters
        params = struct('dt', 0.02,...                  % aquisition time interval [sec]
            'minNumPoints',5,...          % (frames) threshold below which trajectories are discarded
            'minNumPointsLongTraj',5,...   % minimal number of frames to be considered as non confined long trajectory
            'numMSDpoints',5,...           % radius (number of points) to consider in the moving window MSD. [-numPoints numPoints] around each point
            'inspectTrajectories',true,... % allows to inspect visually all, confined and unconfined trajectories
            'plotAssociationDissociationHist',true,...% show the association dissociation histogram
            'plotTrajectories',true,...    % show all 3d trajectories  [true/false]
            'plotTrajDurationHist',true,...% plot trajectory duration histogram [true/false]
            'plotStatisticalParameters',true,... % show the distribution of statistical parameters used for classtering
            'classifyUsingAlpha', true,...    %use anomalous exponent in classification
            'classifyUsingDriftNorm',true,... % use drift norm in classification
            'classifyUsingDiffusion',true,... % use diffusion constant in classification
            'classifyUsingLc', true,...    % use Lc in classification
            'classifyUsingSpringConst', false,... % use Kc in classification
            'numberOfClasses',2,...        % number of classes (confined/unconfined/others..)
            'exportResults', true,...      % export results  [true/false]
            'exportFigures',true,...       % export fig files
            'exportCI', true, ...         % export confidence interval
            'resultFolder',fullfile(pwd,'..','Results'),... % result folder (default=code folder/../)
            'exportPDBtrajectories', true,... % export trajectories as pdb file
            'allowFrameSkipping', true,... % exclude trajectories with frame skipping
            'numHistFitTrials',5,...          % number histogram fitting trials
            'datasetName','',...           % dataset file name
            'datasetPath',fullfile(pwd, '..', 'data', 'tracks'));            % dataset file path
    end

    properties (Access=private)
        handles % graphical handles
        inspectAxesBoundaries = struct('all',[],...
            'confined',[],...
            'unconfined',[]); % get axes limits for inspection of trajectories
        %
        colors = struct('confined',[0, 0.447058823529412, 0.741176470588235],...
            'unconfined',[0.85, 0.33, 0.1]) % 20ms
        % 'unconfined',[0.30, 0.75, 0.93]) % 500ms
    end

    methods

        function obj = AnalyzeTrackingData_withDirection_master(varargin)

            vIn = varargin(:);

            disp('Parsing inputs')
            obj.ParseInputParams(vIn);

            disp('Loading data')
            obj.LoadData;

            disp('Create result folder')
            obj.CreateResultFolder;

            disp('Collect trajectories')
            obj.CollectTrajectoryData;

            disp("Classify trajectories")

            obj.ClassifyTrajectories;

            disp("Collect asso disso data")

            obj.CollectAssociationDissociationData;

            obj.ConstructAssociationDissociationHist;

            disp("Classify confined unconfined trajectories")

            obj.ClassifyConfinedUnconfinedTrajectories;

            % obj.PlotTrajectoriesDurationHist;


            disp("Plotting ...")
            obj.PlotAllTrajectories;
            %
            % obj.PlotAllTrajectorieswithangle;
            %
            % obj.ElevationAnglehistograms;
            %
            % obj.PlotAllTrajectorieswithAzimuthalAngle;
            %
            % obj.AzimuthalAnglehistograms;
            %
            % obj.PlotUnconfinedTrajectories;
            %
            % obj.PlotAllTrajectoriesUnconfinedConfined;
            %
            % obj.InspectTrajectories;
            %
            % obj.DisplaySpatialDistribution;

            obj.PlotStatisticalParameters;
            obj.PlotDisplacementHistogram;
            %
            % obj.ConcatenateAndRandomizeTrajectories;

            disp("Exporting ...")
            obj.ExportResults;
            %
        end

        function LoadData(obj)
            if isempty(obj.params.datasetName)
                % Allow selecting multiple files
                 csvFiles = dir(fullfile(obj.params.datasetPath, '*trackPositions.csv'));
                 obj.params.datasetName = {csvFiles.name};              
            end
            % Sequentally read data files
            dataTemp       = table();
            dataIdx        = [];
            obj.datasetIdx = [];
            maxTrajInd = 0;
            if iscell(obj.params.datasetName)==false % for a single file loaded
                dataTemp = readtable(fullfile(obj.params.datasetPath,obj.params.datasetName));
                varNames = dataTemp.Properties.VariableNames;
                for nIdx = 1:numel(varNames); dataTemp.Properties.VariableNames{nIdx}= ['Var' num2str(nIdx)]; end
                %                   maxTrajInd = max(dataTemp.Var1);
                %                   dataIdx = [size(dataTemp,1)];
                obj.datasetIdx = ones(numel(dataTemp.Var1),1);
            else
                for dIdx = 1:length(obj.params.datasetName)
                    r = readtable(fullfile(obj.params.datasetPath,obj.params.datasetName{dIdx}));
                    varNames = r.Properties.VariableNames;
                    for nIdx = 1:numel(varNames); r.Properties.VariableNames{nIdx}= ['Var' num2str(nIdx)]; end
                    r.Var1 = r.Var1+maxTrajInd;
                    maxTrajInd = max(r.Var1);
                    % rearrange trajectory indices
                    dataTemp = [dataTemp; r];
                    dataIdx  = [dataIdx;size(dataTemp,1)]; % dataset index in file list
                    obj.datasetIdx = [obj.datasetIdx; dIdx*ones(numel(r.Var1),1)];
                end
            end

            obj.data       = dataTemp;

        end

        function PlotAssociationDissociationHistograms(obj,varargin)
            % Association time histogram
            %           cConfined = [0 0.447058823529412 0.741176470588235];
            %           cUnconfined = [0.850980392156863 0.325490196078431 0.0980392156862745];
            if obj.params.plotAssociationDissociationHist
                aHist               = obj.associationTime.histogram;
                binsAssociationHist = obj.associationTime.bins;
                associationFit      = obj.associationTime.fit;
                associationStats    = obj.associationTime.fitStats;

                fig = figure('Units','norm');
                subplot1 = subplot(1,2,1);
                if ~isempty(associationFit)
                    bar(subplot1, binsAssociationHist,aHist,'HandleVisibility','off',...
                        'FaceColor',obj.colors.unconfined,...
                        'EdgeColor',obj.colors.unconfined),hold on
                    plot(subplot1, binsAssociationHist, associationFit(binsAssociationHist),...
                        'LineWidth',3,...
                        'Color','k',...
                        'DisplayName',sprintf('%0.2f%s%0.2f%s',associationFit.a,'exp(-', associationFit.b,'t)'))

                    xlabel(subplot1, 'Time (sec)'),
                    ylabel(subplot1,'Probability'),
                    title(subplot1,'Association')
                    set(subplot1,'FontSize',24,'FontName','Arial','LineWidth',3,'Box','off','Units','norm')
                    leg1 = legend(get(subplot1,'Children'));
                    set(leg1,'Box','off');
                    % add R square annotation
                    annotation(gcf,'textbox',[0.31 0.75 0.06 0.062],...
                        'String',{['R^2=' num2str(associationStats.rsquare)]},...
                        'FontName','Arial',...
                        'FontSize',24,...
                        'EdgeColor','none',...
                        'FitBoxToText','on');
                end
                dHist                = obj.dissociationTime.histogram;
                binsDissociationHist = obj.dissociationTime.bins;
                dissociationFit      = obj.dissociationTime.fit;
                dissociationStats    = obj.dissociationTime.fitStats;

                % Dissociation histogram
                subplot2 =subplot(1,2,2);
                bar(subplot2, binsDissociationHist,dHist,'handleVisibility','off',...
                    'EdgeColor',obj.colors.confined,...
                    'FaceColor',obj.colors.confined), hold on
                if ~isempty(dissociationFit)
                    plot(subplot2, binsDissociationHist, dissociationFit(binsDissociationHist),...
                        'LineWidth',3,...
                        'Color','k',...
                        'DisplayName',sprintf('%0.2f%s%0.2f%s',dissociationFit.a,'exp(-', dissociationFit.b,'t)'))
                    xlabel('Time (sec)')
                    ylabel('Probability')
                    title('Dissociation')
                    set(subplot2,'FontSize',24,'FontName','Arial','LineWidth',3,'Box','off')
                    leg2 = legend(get(subplot2,'Children'));
                    set(leg2,'Box','off');
                    % add R square annotation
                    annotation(gcf,'textbox',[0.76 0.75 0.06 0.062],...
                        'String',{['R^2=' num2str(dissociationStats.rsquare)]},...
                        'FontName','Arial',...
                        'FontSize',24,...
                        'EdgeColor','none',...
                        'FitBoxToText','on');
                    % Match y axes limit to reasonable view
                    ylim = sum([get(subplot2,'YLim'); get(subplot1,'YLim')],1)./2;
                    set(subplot1,'YLim',ylim);
                    set(subplot2,'YLim',ylim);
                end


                % export fig  files
                if obj.params.exportFigures
                    savefig(fig,fullfile(obj.params.figureFolderName,'AssociationDissociationHistogram.fig'))
                end
            end
        end

        function PlotAllTrajectorieswithangle(obj)

            driftConfined = zeros(numel(obj.confined),3);
            for pIdx =1:numel(obj.confined); driftConfined(pIdx,:) = obj.confined(pIdx).drift;end
            driftUnconfined = zeros(numel(obj.unconfined),3);
            for pIdx =1:numel(obj.unconfined); driftUnconfined(pIdx,:) = obj.unconfined(pIdx).drift;end

            [azimuthalangleConfined, elevationangleConfined, rConfined] = cart2sph(driftConfined(:,1),driftConfined(:,2),driftConfined(:,3));
            [azimuthalangleUnconfined, elevationangleUnconfined, rUnconfined] = cart2sph(driftUnconfined(:,1),driftUnconfined(:,2),driftUnconfined(:,3));

            elevationangleConfined = round(elevationangleConfined*1000)/1000; % vector of elevation angles
            elevationangleUnconfined = round(elevationangleUnconfined*1000)/1000; % vector of elevation angles

            cm = colormap(parula(max(3142))); % Angle colourmap

            figtrajelevationangle = figure('Name','Trajectories with elevation angle','Units','norm'); hold on
            for tIdx = 1:numel(obj.confined)
                plot3(obj.confined(tIdx).x,obj.confined(tIdx).y,obj.confined(tIdx).z,...
                    'LineWidth',2, 'Color', cm((round(elevationangleConfined(tIdx)*1000)+1571),:))
            end

            for tIdx = 1:numel(obj.unconfined)
                plot3(obj.unconfined(tIdx).x,obj.unconfined(tIdx).y,obj.unconfined(tIdx).z,...
                    'LineWidth',2, 'Color', cm((round(elevationangleUnconfined(tIdx)*1000)+1571),:))
            end

            cameratoolbar
            daspect([1 1 1])
            xlabel('X (nm)')
            ylabel('Y (nm)')
            zlabel('Z (nm)')
            title('All Trajectories with angle')
            axis tight

            if obj.params.exportFigures
                savefig(figtrajelevationangle,fullfile(obj.params.figureFolderName,'Trajectories_elevation_angle.fig'))
            end

        end

        function ElevationAnglehistograms(obj)
            driftConfined = zeros(numel(obj.confined),3);
            for pIdx =1:numel(obj.confined); driftConfined(pIdx,:) = obj.confined(pIdx).drift;end
            driftUnconfined = zeros(numel(obj.unconfined),3);
            for pIdx =1:numel(obj.unconfined); driftUnconfined(pIdx,:) = obj.unconfined(pIdx).drift;end

            driftAlltraj = [driftConfined; driftUnconfined]; % matrix of drift vectors

            %driftAlltraj = driftUnconfined;

            [azimuthalangle, elevationangle, r] = cart2sph(driftAlltraj(:,1),driftAlltraj(:,2),driftAlltraj(:,3));

            fighist = figure('Name','Elevation angle histogram');
            histogram(elevationangle, 100)
            xlabel('Elevation angle (radians)')
            ylabel('Frequency')

            figpolarhist = figure('Name','Elevation angle polar histogram');
            polarhistogram(elevationangle, 100)

            % Use starting z value for other axis on 2D histogram

            zConfined = zeros(numel(obj.confined),1);
            for pIdx =1:numel(obj.confined); zConfined(pIdx) = obj.confined(pIdx).z(1);end
            zUnconfined = zeros(numel(obj.unconfined),1);
            for pIdx =1:numel(obj.unconfined); zUnconfined(pIdx) = obj.unconfined(pIdx).z(1);end

            z = [zConfined; zUnconfined];
            %z=zUnconfined;

            X = [elevationangle,z];

            figzhist=figure('Name','Elevation angle z histogram');

            hist3(X, [30,30],'LineStyle','none'),
            xlabel('Elevation angle (radians)')
            ylabel('Starting z (nm)')
            zlabel('Frequency')
            title('z elevation angle histogram')
            colorbar
            set(gca,'FontSize',24,'FontName','Arial');
            set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');
            if obj.params.exportFigures
                savefig(fighist,fullfile(obj.params.figureFolderName,'Elevationangle_histogram.fig'))
                savefig(figzhist,fullfile(obj.params.figureFolderName,'Elevationangle_z_histogram.fig'))
                savefig(figpolarhist,fullfile(obj.params.figureFolderName,'Elevationangle_z_polar_histogram.fig'))
                save(fullfile(obj.params.figureFolderName,'Elevation_angle'),'elevationangle')
            end
        end

        function PlotAllTrajectorieswithAzimuthalAngle(obj)
            driftConfined = zeros(numel(obj.confined),3);
            for pIdx =1:numel(obj.confined); driftConfined(pIdx,:) = obj.confined(pIdx).drift;end
            driftUnconfined = zeros(numel(obj.unconfined),3);
            for pIdx =1:numel(obj.unconfined); driftUnconfined(pIdx,:) = obj.unconfined(pIdx).drift;end

            [azimuthalangleConfined, elevationangleConfined, rConfined] = cart2sph(driftConfined(:,1),driftConfined(:,2),driftConfined(:,3));
            [azimuthalangleUnconfined, elevationangleUnconfined, rUnconfined] = cart2sph(driftUnconfined(:,1),driftUnconfined(:,2),driftUnconfined(:,3));

            %azimuthalangleConfined = round(azimuthalangleConfined*1000)/1000; % vector of elevation angles
            %azimuthalangleUnconfined = round(azimuthalangleUnconfined*1000)/1000; % vector of elevation angles

            cm = colormap(parula(max(6284))); % Angle colourmap

            figtrajazimuthalangle = figure('Name','Trajectories with azimuthal angle','Units','norm'); hold on
            for tIdx = 1:numel(obj.confined)
                plot3(obj.confined(tIdx).x,obj.confined(tIdx).y,obj.confined(tIdx).z,...
                    'LineWidth',2, 'Color', cm((round(azimuthalangleConfined(tIdx)*1000)+3142),:))
            end

            for tIdx = 1:numel(obj.unconfined)
                plot3(obj.unconfined(tIdx).x,obj.unconfined(tIdx).y,obj.unconfined(tIdx).z,...
                    'LineWidth',2, 'Color', cm((round(azimuthalangleUnconfined(tIdx)*1000)+3142),:))
            end

            cameratoolbar
            daspect([1 1 1])
            xlabel('X (nm)')
            ylabel('Y (nm)')
            zlabel('Z (nm)')
            zlim([-10000, 10000])
            title('All Trajectories with azimuthal angle')
            axis tight

            if obj.params.exportFigures
                savefig(figtrajazimuthalangle,fullfile(obj.params.figureFolderName,'Trajectories_azimuthal_angle.fig'))
            end
        end

        function AzimuthalAnglehistograms(obj)

            driftConfined = zeros(numel(obj.confined),3);
            for pIdx =1:numel(obj.confined); driftConfined(pIdx,:) = obj.confined(pIdx).drift;end
            driftUnconfined = zeros(numel(obj.unconfined),3);
            for pIdx =1:numel(obj.unconfined); driftUnconfined(pIdx,:) = obj.unconfined(pIdx).drift;end

            driftAlltraj = [driftConfined; driftUnconfined]; % matrix of drift vectors

            [azimuthalangle, elevationangle, r] = cart2sph(driftAlltraj(:,1),driftAlltraj(:,2),driftAlltraj(:,3));

            fighist = figure('Name','Azimuthal angle histogram');
            histogram(azimuthalangle, 100)
            xlabel('Azimuthal angle (radians)')
            ylabel('Frequency')

            figpolarhist = figure('Name','Azimuthal angle polar histogram');
            polarhistogram(azimuthalangle, 100)

            % Use starting z value for other axis on 2D histogram

            zConfined = zeros(numel(obj.confined),1);
            for pIdx =1:numel(obj.confined); zConfined(pIdx) = obj.confined(pIdx).z(1);end
            zUnconfined = zeros(numel(obj.unconfined),1);
            for pIdx =1:numel(obj.unconfined); zUnconfined(pIdx) = obj.unconfined(pIdx).z(1);end

            z = [zConfined; zUnconfined];

            X = [azimuthalangle,z];

            figzhist = figure('Name','Azimuthal angle z histogram');

            hist3(X, [30,30],'LineStyle','none'),
            xlabel('Azimuthal angle (radians)')
            ylabel('Starting z (nm)')
            zlabel('Frequency')
            title('z azimuthal angle histogram')
            colorbar
            set(gca,'FontSize',24,'FontName','Arial');
            set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');

            if obj.params.exportFigures
                savefig(fighist,fullfile(obj.params.figureFolderName,'Azimuthalangle_histogram.fig'))
                savefig(figzhist,fullfile(obj.params.figureFolderName,'Azimuthalangle_z_histogram.fig'))
            end
        end

        function PlotUnconfinedTrajectories(obj,varargin)
            % plot all non-confined trajectories in 3D
            figure('Units','norm'), hold on
            for tIdx = 1:numel(obj.unconfined)
                plot3(obj.unconfined(tIdx).x,obj.unconfined(tIdx).y,obj.unconfined(tIdx).z,...
                    'LineWidth',2)
            end
            cameratoolbar
            daspect([1 1 1])
            xlabel('X (nm)')
            ylabel('Y (nm)')
            zlabel('Z (nm)')
            title('Unconfined trajectories')
            axis tight
        end

        function PlotAllTrajectories(obj,varargin)

            % show all 3D trajectories
            if obj.params.plotTrajectories
                obj.handles.figTraj3d = ...
                    figure('Name','3dtrajectories','NextPlot','Add');
                obj.handles.axTraj3d  =...
                    axes('Parent',obj.handles.figTraj3d,'NextPlot','Add');



                cameratoolbar

                h = obj.handles.figTraj3d;
                clf

                % Step over the individual fields of view
                for i = unique([obj.allTraj.datasetIdx])

                    current_idx = [obj.allTraj.datasetIdx] == i;

                    x = {obj.allTraj(current_idx).x};
                    y = {obj.allTraj(current_idx).y};
                    z = {obj.allTraj(current_idx).z};

                    % plot all trajectories
                    clf
                    hold on

                    xlabel('X (nm)')
                    ylabel('Y (nm)');
                    zlabel('Z (nm)');

                    cellfun(@(x,y,z) plot3(x,y,z,'LineWidth',2),x,y,z)

                    hold off

                    if obj.params.exportFigures

                        savefig(obj.handles.figTraj3d,...
                            fullfile(obj.params.figureFolderName,...
                            ['All_trajectories3D_', num2str(i),'.fig']))
                    end

                end

            end

        end


        function PlotAllTrajectoriesUnconfinedConfined(obj,varargin)

            driftConfined = zeros(numel(obj.confined),3);
            for pIdx =1:numel(obj.confined); driftConfined(pIdx,:) = obj.confined(pIdx).drift;end
            driftUnconfined = zeros(numel(obj.unconfined),3);
            for pIdx =1:numel(obj.unconfined); driftUnconfined(pIdx,:) = obj.unconfined(pIdx).drift;end

            figconfinedunconfined = figure('Name','Confined and Unconfined Trajectories','Units','norm'); hold on

            cameratoolbar

            % Step over the individual fields of view
            for i = unique([[obj.confined.datasetIdx] [obj.unconfined.datasetIdx]])

                current_confined_idx = [obj.confined.datasetIdx] == i;
                current_unconfined_idx = [obj.unconfined.datasetIdx] == i;

                %NB: ALL SUBTRAJECTORIES SHOULD BE RECONNECTED IN THE
                %FUTURE
                % stanley strawbridge 20220405

                clf
                hold on

                % plot all confined subtrajectories
                cellfun(@(x,y,z) plot3(x,y,z,'LineWidth',2, 'Color', obj.colors.confined),...
                    {obj.confined(current_confined_idx).x},...
                    {obj.confined(current_confined_idx).y},...
                    {obj.confined(current_confined_idx).z})

                % plot all unconfined subtrajectories
                cellfun(@(x,y,z) plot3(x,y,z,'LineWidth',2, 'Color', obj.colors.unconfined),...
                    {obj.unconfined(current_unconfined_idx).x},...
                    {obj.unconfined(current_unconfined_idx).y},...
                    {obj.unconfined(current_unconfined_idx).z})

                daspect([1 1 1])

                xlabel('X (nm)')
                ylabel('Y (nm)')
                zlabel('Z (nm)')
                title('All Trajectories Unconfined and Confined')

                axis tight

                hold off

                if obj.params.exportFigures

                    saveName = ['Trajectories_confined_unconfined_', num2str(i),'.fig'];
                    savePath = fullfile(obj.params.figureFolderName,saveName);
                    savefig(figconfinedunconfined,savePath);

                end

            end

        end


        function PlotTrajectoriesDurationHist(obj,varargin)
            % Histogram of the total duration of all trajectories
            if obj.params.plotTrajDurationHist
                figure('Units','norm'),
                nFrames =[obj.allTraj.numFrames];
                hist(nFrames.*obj.params.dt,0.1.*numel(nFrames)),% num hist bin=10% of the number of frames
                xlabel('Trajectory duration (s)'),
                ylabel('Frequency')
                title(obj.params.datasetName)
                set(gca,'FontSize',24,'FontName','Arial','LineWidth',2,'Box','off')
            end
        end

        function InspectTrajectories(obj)
            if obj.params.inspectTrajectories
                obj.InspectAllTrajectories;

                obj.InspectConfinedTrajectories

                obj.InspectUnconfinedTrajectories
            end
        end

        function InspectAllTrajectories(obj)
            % Browse through trjectories and classification of time point
            % into confined and non-confined states
            obj.handles.trajFig    = figure('Units','norm',...
                'WindowButtonMotionFcn',@obj.MouseOverFunction);
            obj.handles.trajSlider = uicontrol('Style','slider',...
                'Min',1,...
                'Max',obj.numTraj,...
                'Value',1,...
                'Units','norm',...
                'Position',[0.1, 0.1, 0.3, 0.05],...
                'callback',@obj.TrajSliderMovement,...
                'SliderStep',[1/obj.numTraj, 1/obj.numTraj]);
            obj.handles.subplot1 = subplot(1,2,1,'Parent',obj.handles.trajFig,'Units','norm');
            obj.handles.subplot2 = subplot(1,2,2,'Parent',obj.handles.trajFig,'Units','norm','YLim',[0 1]);
            set(obj.handles.subplot1,...
                'XLimMode','manual',...
                'YLimMode','manual',...
                'ZLimMode','manual',...
                'XLim',obj.inspectAxesBoundaries.all(1,:),...
                'YLim',obj.inspectAxesBoundaries.all(2,:),...
                'ZLim',obj.inspectAxesBoundaries.all(3,:),...
                'Units','norm');
            obj.PlotIndividualTrajectories(1);% start frame
        end

        function InspectUnconfinedTrajectories(obj)
            % Browse through non-confined trjectories
            obj.handles.unconfinedFig    = figure('Units','norm');
            obj.handles.unconfinedSlider = uicontrol('Style','slider',...
                'Min',1,...
                'Max',numel(obj.unconfined),...
                'Value',1,...
                'Units','norm',...
                'Position',[0.1, 0.1, 0.2, 0.05],...
                'callback',@obj.UnconfinedTrajSliderMovement,...
                'SliderStep',[1/numel(obj.unconfined), 1/numel(obj.unconfined)]);
            obj.handles.unconfinedAx = axes('Parent',obj.handles.unconfinedFig);
            if numel(obj.unconfined)>1
                set(obj.handles.unconfinedAx,...
                    'XLim',obj.inspectAxesBoundaries.unconfined(1,:),...
                    'YLim',obj.inspectAxesBoundaries.unconfined(2,:),...
                    'ZLim',obj.inspectAxesBoundaries.unconfined(3,:));
            end
            obj.PlotIndividualUnconfinedTrajectories(1);% start frame
            cameratoolbar
        end

        function InspectConfinedTrajectories(obj)
            % Browse through confined trjectories
            obj.handles.confinedFig    = figure('Units','norm');
            obj.handles.confinedSlider = uicontrol('Style','slider',...
                'Min',1,...
                'Max',numel(obj.confined),...
                'Value',1,...
                'Units','norm',...
                'Position',[0.1, 0.1, 0.2, 0.05],...
                'callback',@obj.ConfinedTrajSliderMovement,...
                'SliderStep',[1/numel(obj.confined), 1/numel(obj.confined)]);
            obj.handles.confinedAx = axes('Parent',obj.handles.confinedFig);
            if numel(obj.confined)>1
                set(obj.handles.confinedAx,...
                    'XLim',obj.inspectAxesBoundaries.confined(1,:),...
                    'YLim',obj.inspectAxesBoundaries.confined(2,:),...
                    'ZLim',obj.inspectAxesBoundaries.confined(3,:))
            end
            obj.PlotIndividualConfinedTrajectories(1);% start frame
            cameratoolbar
        end

        function CreateResultFolder(obj)
            if iscell(obj.params.datasetName)==false
                obj.params.resultFolderName = fullfile(obj.params.resultFolder,obj.params.datasetName);
                [~,~] = mkdir(obj.params.resultFolderName);
            else
                c    = clock;
                obj.params.resultFolderName =fullfile(obj.params.resultFolder,[num2str(c(3)),'_',num2str(c(2)),'_',num2str(c(1)),'_',num2str(c(4)),'_',num2str(c(5))]);
                [~,~] = mkdir(obj.params.resultFolderName);
            end

            % Generate folder heirarchy in result folder
            % create PDB folder
            obj.params.pdbPath = fullfile(obj.params.resultFolderName,'PDB');
            [~,~]              = mkdir(obj.params.pdbPath);
            % Create confined folder
            obj.params.resultFolderConfined = fullfile(obj.params.resultFolderName,'Confined');
            [~,~]                  = mkdir(obj.params.resultFolderConfined);
            % Create result folder unconfined
            obj.params.resultFolderUnconfined = fullfile(obj.params.resultFolderName,'Unconfined');
            [~,~]                  = mkdir(obj.params.resultFolderUnconfined);
            % Create result folder for all trajectories
            obj.params.resultFolderAll = fullfile(obj.params.resultFolderName,'AllTrajectories');
            [~,~]                  = mkdir(obj.params.resultFolderAll);
            % Create trajectory data
            obj.params.allTrajectoryFolderName = fullfile(obj.params.resultFolderAll,'TrajectoryData');
            [~,~]            = mkdir(obj.params.allTrajectoryFolderName);
            % Create unconfined trajectory data folder
            obj.params.unconfinedTrajectoryFolderName = fullfile(obj.params.resultFolderUnconfined,'TrajectoryData');
            [~,~]            = mkdir(obj.params.unconfinedTrajectoryFolderName);
            % Create confined trajectory dataPath
            obj.params.confinedTrajectoryFolderName = fullfile(obj.params.resultFolderConfined,'TrajectoryData');
            [~,~]            = mkdir(obj.params.confinedTrajectoryFolderName);
            % Create Association folder
            obj.params.associationFolderName = fullfile(obj.params.resultFolderName,'Association');
            [~,~]            = mkdir(obj.params.associationFolderName);
            % Create Dissociation folder
            obj.params.dissociationFolderName = fullfile(obj.params.resultFolderName,'Dissociation');
            [~,~]             = mkdir(obj.params.dissociationFolderName);
            % create figure folder
            obj.params.figureFolderName = fullfile(obj.params.resultFolderName,'Figures');
            [~,~]             = mkdir(obj.params.figureFolderName);
        end



        function ExportResults(obj)
            if obj.params.exportResults

                obj.ExportStatisticalParameters;
                obj.ExportTrajectoryDataToCsv;

                % obj.ExportTrajectoriesAsPDB;

                obj.ExportAssociationDissociationData;

                % Export file names
                fid = fopen(fullfile(obj.params.resultFolderName,'FileList.txt'),'w');
                if iscell(obj.params.datasetName)
                    for fnIdx =1:length(obj.params.datasetName)
                        fprintf(fid,'%s\n',obj.params.datasetName{fnIdx});
                    end
                else
                    fprintf(fid,'%s,',obj.params.datasetName);
                end
                fclose(fid);
            end

        end

        function DisplayMovie(obj)% UNFINISHED
            fig = figure; cameratoolbar
            ax  = axes('Parent',fig);
            title(ax,['Frame ', num2str(0)]);
            for fIdx = 1:20000
                % set(ax,'NextPlot','Add')
                p    = [];
                next = 1;
                for tIdx = 1:obj.numTraj
                    tInd =(obj.allTraj(tIdx).frames==fIdx);
                    if any(tInd)
                        p(next,1) = obj.allTraj(tIdx).x(tInd);
                        p(next,2) = obj.allTraj(tIdx).y(tInd);
                        p(next,3) = obj.allTraj(tIdx).z(tInd);
                        next     = next+1;
                    end

                end
                if ~isempty(p)
                    plot3(p(:,1),p(:,2), p(:,3),'Marker','o','LineStyle','none')
                    set(gca,'XLim',[-1e5 1e5],'YLim',[-1e5 1e5])
                    title(ax,['Frame', num2str(fIdx)])
                    %set(ax,'NextPlot','ReplaceChildren')
                    drawnow
                end

            end
        end

        function DisplaySpatialDistribution(obj)
            % Spatial distribution of the X Y components from all trajectories
            varnames = obj.data.Properties.VariableNames;
            figure,

            hist3([obj.data.(varnames{3}),obj.data.(varnames{4})],[30,30],'LineStyle','none'),
            xlabel('X (nm)')
            ylabel('Y (nm)')
            zlabel('Frequency')
            title('Spatial Dist.')
            colorbar
            set(gca,'FontSize',24,'FontName','Arial');
            set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');

        end

        function ConcatenateAndRandomizeTrajectories(obj)
            % compute the mean-squared displacement of confined and
            % unconfined trajectories.
            % concatenate all confined/unconfined trajectoeries and
            % randomize the concatenation order to generate an ensemble
            % for which we compute the MSD

            numExp = 100; % number of times to randomize the order of concatenation
            ft     = fittype('b*x.^a'); % quadratic model to fit the data
            % unconfined trajectories

            sdUnconfined = []; % squared displacement unconfined traj.
            for expIdx = 1:numExp
                zUnconf= [0 0 0];
                for zIdx = randperm(numel(obj.unconfined))
                    temp = [obj.unconfined(zIdx).x-obj.unconfined(zIdx).x(1)+ zUnconf(end,1),...
                        obj.unconfined(zIdx).y-obj.unconfined(zIdx).y(1)+ zUnconf(end,2),...
                        obj.unconfined(zIdx).z-obj.unconfined(zIdx).z(1)+ zUnconf(end,3)];
                    zUnconf = [zUnconf;temp(2:end,:)]; % squared displacement
                end
                % zUnconf      = zUnconf; % remove ./1000, work in µm
                sdUnconfined = [sdUnconfined, sum(zUnconf.^2,2)];
            end
            msdUnconfined  = mean(sdUnconfined,2);
            maxIndUnconf   = numel(msdUnconfined); % max index to fit the MSD
            timeUnconfined = (0:numel(msdUnconfined) -1).*obj.params.dt;
            fitUnconfined  = fit(timeUnconfined(1:maxIndUnconf)',msdUnconfined(1:maxIndUnconf),ft,...
                'StartPoint',rand(1,2),'Lower',[0 0]);

            sdConfined = []; % squared-displacement for confined traj.
            for expIdx = 1:numExp
                zConf = [0 0 0];
                for zIdx = randperm(numel(obj.confined))
                    temp = [obj.confined(zIdx).x-obj.confined(zIdx).x(1)+ zConf(end,1),...
                        obj.confined(zIdx).y-obj.confined(zIdx).y(1)+ zConf(end,2),...
                        obj.confined(zIdx).z-obj.confined(zIdx).z(1)+ zConf(end,3)];
                    zConf = [zConf;temp(2:end,:)];
                end
                % zConf      = zConf./1000; % remove to work in µm
                sdConfined = [sdConfined, sum(zConf.^2,2)];
            end
            msdConfined    = mean(sdConfined,2);
            timeConfined   = (0:numel(msdConfined)-1).*obj.params.dt;
            maxIndConfined = numel(msdConfined);
            fitConfined    = fit(timeConfined(1:maxIndConfined)',msdConfined(1:maxIndConfined),ft,...
                'StartPoint',rand(1,2),'Lower',[0 0]);

            % plot MSD and fit Confined
            figure('Name','MSD confined','Units','norm'), hold on
            plot(timeConfined(1:maxIndConfined), msdConfined(1:maxIndConfined),'LineWidth',5)
            plot(timeConfined(1:maxIndConfined),ft(fitConfined.a,fitConfined.b,timeConfined(1:maxIndConfined)),...
                'LineWidth',5,'LineStyle','--','DisplayName','fit confined');
            axis tight
            title('MSD confined')
            xlabel('Time (sec)')
            ylabel('MSD_u(\mu m^2)')
            text(10,max(msdConfined(1:maxIndConfined))/2,['MSD_c(t)=', num2str(fitConfined.b),'t^{', num2str(fitConfined.a),'}'],'Parent',gca,'FontSize',20)
            set(gca,'LineWidth',3,'FontSize',30,'FontName','Arial','Units','norm')

            % plot MSD and fit Unconfined
            figure('Name','MSD unconfined','Units','norm'), hold on
            plot(timeUnconfined(1:maxIndUnconf),msdUnconfined(1:maxIndUnconf),'LineWidth',5)
            plot(timeUnconfined(1:maxIndUnconf),ft(fitUnconfined.a,fitUnconfined.b,timeUnconfined(1:maxIndUnconf)),...
                'LineWidth',5,'LineStyle','--','DisplayName','fit unconfined');

            % plot pure diffusion case
            %             plot(timeConfined(1:maxIndConfined),timeConfined(1:maxIndConfined),'LineWidth',5,'DisplayName','pure diffusion')

            axis tight
            title('MSD unconfined')
            xlabel('Time (sec)')
            ylabel('MSD_u(\mu m^2)')
            text(10,max(msdUnconfined(1:maxIndUnconf))/2,['MSD_u(t)=',num2str(fitUnconfined.b),'t^{',num2str(fitUnconfined.a),'}'],'Parent',gca,'FontSize',20);
            set(gca,'LineWidth',3,'FontSize',30,'FontName','Arial','Units','norm')

        end

        function PlotStatisticalParameters(obj)
            % plot statistical parameters used for classification
            if obj.params.plotStatisticalParameters
                obj.PlotDrift
                %                 obj.PlotDriftwithangle
                %                 obj.PlotDriftwithAzimuthalAngle
                obj.PlotDiffusion
                obj.PlotLc
                %             obj.PlotKc
                obj.PlotAlpha
            end
        end

        function PlotDrift(obj)
            %             cConfined = [0 0.447058823529412 0.741176470588235];
            %             cUnconfined = [0.850980392156863 0.325490196078431 0.0980392156862745];

            driftConfined = zeros(numel(obj.confined),3);
            for pIdx =1:numel(obj.confined)

                driftConfined(pIdx,:) = obj.confined(pIdx).drift;
                obj.confined(pIdx).driftMagnitude = sqrt(sum(driftConfined(pIdx,:).^2,2));
            end
            driftUnconfined = zeros(numel(obj.unconfined),3);
            for pIdx =1:numel(obj.unconfined)
                driftUnconfined(pIdx,:) = obj.unconfined(pIdx).drift;
                obj.unconfined(pIdx).driftMagnitude = sqrt(sum(driftUnconfined(pIdx).^2,2));

            end
            normDriftConfined   = sqrt(sum(driftConfined.^2,2));
            normDriftUnconfined = sqrt(sum(driftUnconfined.^2,2));


            figDriftNorm3D = figure('Name', 'Drift vectors');
            axDriftNorm3D  = axes('Parent',figDriftNorm3D);
            plot3(axDriftNorm3D ,driftConfined(:,1),driftConfined(:,2), driftConfined(:,3),'o',...
                'DisplayName','Confined',...
                'Color',obj.colors.confined,...
                'MarkerFaceColor',obj.colors.confined)
            hold on,
            plot3(axDriftNorm3D ,driftUnconfined(:,1), driftUnconfined(:,2),driftUnconfined(:,3),'o',...
                'DisplayName','Unconfined',...
                'Color',obj.colors.unconfined,...
                'MarkerFaceColor',obj.colors.unconfined)
            zlabel(axDriftNorm3D,'Drift Z','FontSize',20,'FontName','Arial')
            ylabel(axDriftNorm3D,'Drift Y','FontSize',20,'FontName','Arial')
            xlabel(axDriftNorm3D ,'Drift X','FontSize',20,'FontName','Arial')
            set(axDriftNorm3D ,'LineWidth',3,'FontSize',20,'FontName','Arial')
            cameratoolbar

            if obj.params.exportFigures
                savefig(figDriftNorm3D,fullfile(obj.params.figureFolderName,'DriftNorm3D.fig'))
            end
            % plot histograms
            % first compute the histogram for the two sets together
            figTemp = figure;
            ht = histogram([normDriftConfined;normDriftUnconfined],100);
            bw = ht.BinWidth;
            vt = ht.Values;
            bt = ht.BinEdges(1:end-1)+bw/2;
            close(figTemp)

            figDriftNormUnclassified = figure('Name','Drift norm- before classification');
            axDriftNormUnclassified  = axes('Parent',figDriftNormUnclassified);
            bar(axDriftNormUnclassified,bt,vt)
            xlabel(axDriftNormUnclassified,'Drift norm','Fontsize',20,'FontName','Arial')
            ylabel(axDriftNormUnclassified,'Frequency','Fontsize',20,'FontName','Arial')
            set(axDriftNormUnclassified,'LineWidth',3,'FontSize',20,'FontName','Arial')
            % export figure
            if obj.params.exportFigures
                savefig(figDriftNormUnclassified,fullfile(obj.params.figureFolderName,'DriftNormUnclassified.fig'))
            end


            figDriftNormHist = figure('Name','Drift norm- after classification');
            axDriftNormHist  = axes('Parent',figDriftNormHist);
            % then compute the histogram for each set individually
            hc = histogram(normDriftConfined,'BinWidth',bw);
            % compute position of center of bins
            bc = hc.BinEdges(1:end-1)+(hc.BinEdges(2)-hc.BinEdges(1))/2;
            vc = hc.Values;
            clear hc
            hu = histogram(normDriftUnconfined,'BinWidth',bw);
            % compute position of center of bins
            bu = hu.BinEdges(1:end-1)+(hu.BinEdges(2)-hu.BinEdges(1))/2;
            vu = hu.Values;
            clear hu
            bar(axDriftNormHist,bc,vc,'FaceColor',obj.colors.confined,'EdgeColor',obj.colors.confined)
            hold on,
            bar(axDriftNormHist,bu,vu,'FaceColor',obj.colors.unconfined,'EdgeColor',obj.colors.unconfined);
            
            % Add C/U text to figure (Yuze 20250129)
            nCon = length(normDriftConfined);
            nUnc = length(normDriftUnconfined);
            txtConfined = sprintf("C %.1f%% (n=%d)", 100*nCon/(nCon+nUnc), nCon);
            txtUnconfined = sprintf("U %.1f%% (n=%d)", 100*nUnc/(nCon+nUnc), nUnc);
            text(mean(bc), max(vc), txtConfined, "FontSize", 20, 'FontName','Arial', 'Color', obj.colors.confined)
            text(mean(bu), max(vu), txtUnconfined, "FontSize", 20, 'FontName','Arial', 'Color', obj.colors.unconfined)

            xlabel(axDriftNormHist,'Drift norm','Fontsize',20,'FontName','Arial')
            ylabel(axDriftNormHist,'Frequency','Fontsize',20,'FontName','Arial')
            set(axDriftNormHist,'LineWidth',3,'FontSize',20,'FontName','Arial')
            % export figure
            if obj.params.exportFigures
                savefig(figDriftNormHist,fullfile(obj.params.figureFolderName,'DriftNormClassified.fig'))
            end
            %             figure('Name','Drift norm'),
            %             hold on
            %             plot(normDriftConfined,'DisplayName','confined',...
            %                 'Color',obj.colors.confined,'LineWidth',2)
            %             plot(normDriftUnconfined,'DisplayName','unconfined',...
            %                 'Color',obj.colors.unconfined,'LineWidth',2)
            %             xlabel('Trajectory index'), ylabel('||Drift||')
            %             set(gca,'LineWidth',3,'FontSize',20,'FontName','Arial')
        end

        function PlotDriftwithangle(obj)

            driftConfined = zeros(numel(obj.confined),3);
            for pIdx =1:numel(obj.confined); driftConfined(pIdx,:) = obj.confined(pIdx).drift;end
            driftUnconfined = zeros(numel(obj.unconfined),3);
            for pIdx =1:numel(obj.unconfined); driftUnconfined(pIdx,:) = obj.unconfined(pIdx).drift;end

            driftAlltraj = [driftConfined; driftUnconfined]; % matrix of drift vectors

            [azimuthalangle, elevationangle, r] = cart2sph(driftAlltraj(:,1),driftAlltraj(:,2),driftAlltraj(:,3));

            elevationangle = round(elevationangle*1000)/1000; % vector of elevation angles

            cm = colormap(parula(max(3142))); % Angle colourmap

            figdriftangle3d = figure('Name','Drift with elevation angle','NextPlot','Add');
            axdriftangle3d  = axes('Parent',figdriftangle3d,'NextPlot','Add');
            xlabel('X (nm)')
            ylabel('Y (nm)');
            zlabel('Z (nm)');

            for tIdx = 1:(numel(elevationangle))
                plot3(axdriftangle3d,driftAlltraj(tIdx,1),driftAlltraj(tIdx,2), driftAlltraj(tIdx,3),'o',...
                    'DisplayName','All drift vectors with direction',...
                    'Color',cm((round(elevationangle(tIdx)*1000)+1571),:),...
                    'MarkerFaceColor',cm((round(elevationangle(tIdx)*1000)+1571),:))
            end

            zlabel(axdriftangle3d,'Drift Z','FontSize',20,'FontName','Arial')
            ylabel(axdriftangle3d,'Drift Y','FontSize',20,'FontName','Arial')
            xlabel(axdriftangle3d ,'Drift X','FontSize',20,'FontName','Arial')
            set(axdriftangle3d ,'LineWidth',3,'FontSize',20,'FontName','Arial')
            cameratoolbar

            if obj.params.exportFigures
                savefig(figdriftangle3d,fullfile(obj.params.figureFolderName,'Drift_Elevation_Angle.fig'))
            end

        end

        function PlotDriftwithAzimuthalAngle(obj)
            driftConfined = zeros(numel(obj.confined),3);
            for pIdx =1:numel(obj.confined); driftConfined(pIdx,:) = obj.confined(pIdx).drift;end
            driftUnconfined = zeros(numel(obj.unconfined),3);
            for pIdx =1:numel(obj.unconfined); driftUnconfined(pIdx,:) = obj.unconfined(pIdx).drift;end

            driftAlltraj = [driftConfined; driftUnconfined]; % matrix of drift vectors

            [azimuthalangle, elevationangle, r] = cart2sph(driftAlltraj(:,1),driftAlltraj(:,2),driftAlltraj(:,3));

            %azimuthalangle = round(azimuthalangle*1000)/1000; % vector of elevation angles

            cm = colormap(parula(max(6284))); % Angle colourmap

            figdriftangle3d = figure('Name','Drift with azimuthal angle','NextPlot','Add');
            axdriftangle3d  = axes('Parent',figdriftangle3d,'NextPlot','Add');
            xlabel('X (nm)')
            ylabel('Y (nm)');
            zlabel('Z (nm)');

            for tIdx = 1:(numel(azimuthalangle))
                plot3(axdriftangle3d,driftAlltraj(tIdx,1),driftAlltraj(tIdx,2), driftAlltraj(tIdx,3),'o',...
                    'DisplayName','All drift vectors with direction',...
                    'Color',cm((round(azimuthalangle(tIdx)*1000)+3142),:),...
                    'MarkerFaceColor',cm((round(azimuthalangle(tIdx)*1000)+3142),:))
            end

            zlabel(axdriftangle3d,'Drift Z','FontSize',20,'FontName','Arial')
            ylabel(axdriftangle3d,'Drift Y','FontSize',20,'FontName','Arial')
            xlabel(axdriftangle3d ,'Drift X','FontSize',20,'FontName','Arial')
            set(axdriftangle3d ,'LineWidth',3,'FontSize',20,'FontName','Arial')
            cameratoolbar

            if obj.params.exportFigures
                savefig(figdriftangle3d,fullfile(obj.params.figureFolderName,'Drift_Azimuthal_Angle.fig'))
            end
        end

        function PlotDiffusion(obj)


            diffusionConfined   = [obj.confined.diffusionConst];
            diffusionUnconfined = [obj.unconfined.diffusionConst];
            % compute diffusion histogram before classification
            % for both confined and unconfined to obtain a common bin size
            figTemp = figure;
            ht = histogram([diffusionConfined,diffusionUnconfined],100);
            bw = ht.BinWidth;
            vt = ht.Values;
            bt = ht.BinEdges(1:end-1)+bw/2;
            close(figTemp)

            figuDiffHistUnclassified = figure('Name','Diffusion- before classification');
            axDiffHistUnclassified   = axes('Parent', figuDiffHistUnclassified );
            bar(axDiffHistUnclassified, bt,vt);
            xlabel(axDiffHistUnclassified,'Diffusion const','Fontsize',20,'FontName','Arial')
            ylabel(axDiffHistUnclassified,'Frequency','Fontsize',20,'FontName','Arial')
            set(axDiffHistUnclassified,'FontSize',20,'FontName','Arial','LineWidth',2)
            % export figures
            if obj.params.exportFigures
                savefig(figuDiffHistUnclassified,fullfile(obj.params.figureFolderName,'DiffusionConstUnclassified.fig'))
            end

            % plot histograms after classification
            figDiffusionHist = figure('Name','Diffusion-after classification');
            axDiffusionHist  = axes('Parent',figDiffusionHist);

            hc = histogram(diffusionConfined,'BinWidth',bw);
            vc = hc.Values;
            bc = hc.BinEdges(1:end-1)+bw/2;
            hu = histogram(diffusionUnconfined,'BinWidth',bw);
            vu = hu.Values;
            bu = hu.BinEdges(1:end-1)+bw/2;

            bar(axDiffusionHist,bc,vc,'FaceColor',obj.colors.confined,...
                'EdgeColor',obj.colors.confined,...
                'DisplayName','confined')
            hold on,
            bar(axDiffusionHist,bu,vu,...
                'FaceColor',obj.colors.unconfined,...
                'EdgeColor',obj.colors.unconfined,...
                'DisplayName','unconfined');

            % Add C/U text to figure (Yuze 20250129)
            nCon = length(diffusionConfined);
            nUnc = length(diffusionUnconfined);
            txtConfined = sprintf("C %.1f%% (n=%d)", 100*nCon/(nCon+nUnc), nCon);
            txtUnconfined = sprintf("U %.1f%% (n=%d)", 100*nUnc/(nCon+nUnc), nUnc);
            text(mean(bc), max(vc), txtConfined, "FontSize", 20, 'FontName','Arial', 'Color', obj.colors.confined)
            text(mean(bu), max(vu), txtUnconfined, "FontSize", 20, 'FontName','Arial', 'Color', obj.colors.unconfined)

            xlabel(axDiffusionHist ,'Diffusion const','Fontsize',20,'FontName','Arial')
            ylabel(axDiffusionHist ,'Frequency','Fontsize',20,'FontName','Arial')
            set(axDiffusionHist,'FontSize',20,'FontName','Arial','LineWidth',2)
            % export figure
            if obj.params.exportFigures
                savefig(figDiffusionHist,fullfile(obj.params.figureFolderName,'DiffusionContClassified.fig'))
            end
        end

        function PlotKc(obj)
            figTemp      = figure;
            kcConfined   = [obj.confined.Kc];
            kcUnconfined = [obj.unconfined.Kc];
            ht           = histogram([kcConfined;kcUnconfined],100);
            bw           = ht.BinWidth;
            vt           = ht.Values;
            bt           = ht.BinEdges(1:end-1)+bw./2;
            close(figTemp);
            figKcBefore = figure('Name','Kc-Before classification');
            axKcBefore  = axes('Parent',figKcBefore);
            bar(axKcBefore,bt,vt)
            xlabel(axKcBefore,'Spring const, Kc','Fontsize',20,'FontName','Arial')
            ylabel(axKcBefore,'Frequency','Fontsize',20,'FontName','Arial')
            % export figure
            if obj.params.exportFigures
                savefig(figKcBefore,fullfile(obj.params.figureFolderName,'KcUnclassified.fig'))
            end
            figTemp =figure;
            hc = histogram(kcConfined,'BinWidth',bw);
            vc = hc.Values;
            bc = hc.BinEdges(1:end-1)+bw./2;

            hu = histogram(kcUnconfined,'BiinWidth',bw);
            vu = hu.Values;
            bu = hu.BinEdges(1:end-1)+bw./2;
            close(figTemp)

            figKcAfter = figure('Name','Kc- after classification');
            axKcAfter  = axes('Parent',figKcAfter);
            bar(axKcAfter,bc,vc,'FaceColor','b','DisplayName','confined')
            hold on,
            bar(axKcAfter,bu,vu,'FaceColor','r','DisplayName','unconfined');

            % Add C/U text to figure (Yuze 20250129)
            nCon = length(kcConfined);
            nUnc = length(kcUnconfined);
            txtConfined = sprintf("C %.1f%% (n=%d)", 100*nCon/(nCon+nUnc), nCon);
            txtUnconfined = sprintf("U %.1f%% (n=%d)", 100*nUnc/(nCon+nUnc), nUnc);
            text(mean(bc), max(vc), txtConfined, "FontSize", 20, 'FontName','Arial', 'Color', obj.colors.confined)
            text(mean(bu), max(vu), txtUnconfined, "FontSize", 20, 'FontName','Arial', 'Color', obj.colors.unconfined)

            xlabel(axKcAfter,'Spring const, Kc','Fontsize',20,'FontName','Arial')
            ylabel(axKcAfter,'Frequency','Fontsize',20,'FontName','Arial')
            % export figure
            if obj.params.exportFigures
                savefig(figKcAfter,fullfile(obj.params.figureFolderName,'KcClassified.fig'))
            end
        end

        function PlotLc(obj)

            lcConfined = [obj.confined(:).Lc];
            lcUnconfined = [obj.unconfined(:).Lc];
            figTemp = figure();
            ht = histogram([lcConfined,lcUnconfined],100);
            bw = ht.BinWidth;
            vt = ht.Values;
            bt = ht.BinEdges(1:end-1)+bw/2;
            close(figTemp)

            figLcBefore = figure('Name','Lc-before classification');
            axLcBefore = axes('Parent',figLcBefore);
            bar(axLcBefore,bt,vt)
            xlabel(axLcBefore ,'Lc','Fontsize',20,'FontName','Arial')
            ylabel(axLcBefore ,'Frequency','Fontsize',20,'FontName','Arial')
            set(axLcBefore ,'FontSize',20,'FontName','Arial','LineWidth',2);
            % export figure
            if obj.params.exportFigures
                savefig(figLcBefore,fullfile(obj.params.figureFolderName,'LcUnclassified.fig'))
            end

            figLcAfter = figure('Name','Lc- after classification');
            axLcAfter  = axes('Parent',figLcAfter);
            hc         = histogram(lcConfined,'BinWidth',bw);
            bc         = hc.BinEdges(1:end-1)+bw/2;
            vc         = hc.Values;
            hu         = histogram(lcUnconfined,'BinWidth',bw);
            bu         = hu.BinEdges(1:end-1)+bw/2;
            vu         = hu.Values;

            bar(axLcAfter,bc,vc,'FaceColor',obj.colors.confined,...
                'EdgeColor',obj.colors.confined,...
                'DisplayName','confined')

            hold on,
            bar(axLcAfter,bu,vu,'FaceColor',obj.colors.unconfined,...
                'EdgeColor',obj.colors.unconfined,...
                'DisplayName','unconfined');

            % Add C/U text to figure (Yuze 20250129)
            nCon = length(lcConfined);
            nUnc = length(lcUnconfined);
            txtConfined = sprintf("C %.1f%% (n=%d)", 100*nCon/(nCon+nUnc), nCon);
            txtUnconfined = sprintf("U %.1f%% (n=%d)", 100*nUnc/(nCon+nUnc), nUnc);
            text(mean(bc), max(vc), txtConfined, "FontSize", 20, 'FontName','Arial', 'Color', obj.colors.confined)
            text(mean(bu), max(vu), txtUnconfined, "FontSize", 20, 'FontName','Arial', 'Color', obj.colors.unconfined)

            xlabel(axLcAfter,'Lc','Fontsize',20,'FontName','Arial')
            ylabel(axLcAfter,'Frequency','Fontsize',20,'FontName','Arial')
            set(gca,'FontSize',20,'FontName','Arial','LineWidth',2);
            % export figure
            if obj.params.exportFigures
                savefig(figLcAfter,fullfile(obj.params.figureFolderName,'LcClassified.fig'))
            end

        end

        function PlotAlpha(obj)

            aConfined   = [obj.confined(:).alpha];
            aUnconfined = [obj.unconfined(:).alpha];
            figTemp = figure();
            ht = histogram([aConfined, aUnconfined],100);
            bw = ht.BinWidth;
            vt = ht.Values;
            bt = ht.BinEdges(1:end-1)+bw/2;
            close(figTemp)

            figAlphaBefore = figure('Name','Alpha - before classification');
            axAlphaBefore  = axes('Parent',figAlphaBefore);
            bar(axAlphaBefore,bt,vt);
            xlabel(axAlphaBefore,'Anomalous exponent','Fontsize',20,'FontName','Arial')
            ylabel(axAlphaBefore,'Frequency','Fontsize',20,'FontName','Arial')
            set(axAlphaBefore,'FontSize',20,'FontName','Arial','LineWidth',2);
            % export figure
            if obj.params.exportFigures
                savefig(figAlphaBefore,fullfile(obj.params.figureFolderName,'alphaUnclassified.fig'))
            end

            figAlphaAfter = figure('Name','Alpha - after classification');
            axAlphaAfter = axes('Parent',figAlphaAfter);

            hc     = histogram(aConfined,'BinWidth',bw);
            vc     = hc.Values;
            bc     = hc.BinEdges(1:end-1)+bw/2;
            hu     = histogram(aUnconfined,'BinWidth',bw);
            vu     = hu.Values;
            bu     = hu.BinEdges(1:end-1)+bw/2;

            bar(axAlphaAfter ,bc,vc,'FaceColor',obj.colors.confined,...
                'EdgeColor',obj.colors.confined,...
                'DisplayName','confined')
            hold on,
            bar(axAlphaAfter,bu,vu,'FaceColor',obj.colors.unconfined,...
                'EdgeColor',obj.colors.unconfined,...
                'DisplayName','unconfined');

            % Add C/U text to figure (Yuze 20250129)
            nCon = length(aConfined);
            nUnc = length(aUnconfined);
            txtConfined = sprintf("C %.1f%% (n=%d)", 100*nCon/(nCon+nUnc), nCon);
            txtUnconfined = sprintf("U %.1f%% (n=%d)", 100*nUnc/(nCon+nUnc), nUnc);
            text(mean(bc), max(vc), txtConfined, "FontSize", 20, 'FontName','Arial', 'Color', obj.colors.confined)
            text(mean(bu), max(vu), txtUnconfined, "FontSize", 20, 'FontName','Arial', 'Color', obj.colors.unconfined)

            xlabel(axAlphaAfter,'Anomalous exponent','Fontsize',20,'FontName','Arial')
            ylabel(axAlphaAfter,'Frequency','Fontsize',20,'FontName','Arial')
            set(axAlphaAfter,'FontSize',20,'FontName','Arial','LineWidth',2);
            % export figure
            if obj.params.exportFigures
                savefig(figAlphaAfter,fullfile(obj.params.figureFolderName,'alphaClassified.fig'))
            end

        end

        function PlotDisplacementHistogram(obj)
            % compute displacement histogram for confined and unconfined
            % trajectories after classification
            numBins = 200;
            %             dc = []; for cIdx = 1:numel(obj.confined);   dc = [dc;obj.confined(cIdx).displacement];end
            %             du = []; for cIdx = 1:numel(obj.unconfined); du = [du;obj.unconfined(cIdx).displacement];end
            dc = [obj.confined(:).displacement];
            du = [obj.unconfined(:).displacement];
            [hc,bc] = hist(dc,numBins);
            [hu,bu] = hist(du,numBins);
            [ht,bt] = hist([dc,du],numBins);

            nCon = length(dc);
            nUnc = length(du);

            dispFig =figure;
            hold on,
            bar(bc,hc./trapz(bc,hc),'FaceColor',obj.colors.confined,'DisplayName',sprintf("C %.1f%% (n=%d)", 100*nCon/(nCon+nUnc), nCon))
            bar(bu,hu./trapz(bu,hu),'FaceColor',obj.colors.unconfined,'DisplayName',sprintf("U %.1f%% (n=%d)", 100*nUnc/(nCon+nUnc), nUnc))
            plot(bt,ht./trapz(bt,ht),'g','DisplayName','Confined+Unconfined')
            xlabel('|X(t+dt)-X(t)| (mu m)');
            ylabel('Frequency')
            legend(flipud(get(gca,'Children')))
            set(gca,'FontSize',20,'FontName','Arial','LineWidth',2);

            % export figure
            if obj.params.exportFigures
                savefig(dispFig,fullfile(obj.params.figureFolderName,'displacement.fig'))
            end

            % Compute the central chi distribution for the displacement
            % collect the centralized displacement
            dx  = [];dy  = [];dz  = [];
            dxc = [];dyc = [];dzc = [];
            dxu = [];dyu = [];dzu = [];
            for cIdx = 1:numel(obj.allTraj)
                dx = [dx; diff(obj.allTraj(cIdx).x)];
                dy = [dy; diff(obj.allTraj(cIdx).y)];
                dz = [dz; diff(obj.allTraj(cIdx).z)];
            end
            y = sqrt(((dx-mean(dx))/std(dx)).^2 +...
                ((dy-mean(dy))/std(dy)).^2 +...
                ((dz-mean(dz))/std(dz)).^2);
            %               y= sqrt(sum([dx,dy,dz].^2,2));
            %             for cIdx = 1:numel(obj.confined)
            %                dxc = [dxc; diff(obj.confined(cIdx).x)];
            %                dyc = [dyc; diff(obj.confined(cIdx).y)];
            %                dzc = [dzc; diff(obj.confined(cIdx).z)];
            %             end
            %               yc = sqrt(((dxc-0*mean(dxc))/std(dxc)).^2 +...
            %                         ((dyc-0*mean(dyc))/std(dyc)).^2 +...
            %                         ((dzc-0*mean(dzc))/std(dzc)).^2);
            %
            %             for cIdx = 1:numel(obj.unconfined)
            %                dxu = [dxu; diff(obj.unconfined(cIdx).x)];
            %                dyu = [dyu; diff(obj.unconfined(cIdx).y)];
            %                dzu = [dzu; diff(obj.unconfined(cIdx).z)];
            %             end

            %             yu = sqrt(((dxu-0*mean(dxu))/std(dxu)).^2 +...
            %                       ((dyu-0*mean(dyu))/std(dyu)).^2 +...
            %                       ((dzu-0*mean(dzu))/std(dzu)).^2);
            %             figure,
            %             [hn, bn] = hist(y,numBins);
            %             chipdf   = @(x,k) (x.^(k-1) .*exp(-(x.^2)./2))./(2.^(k/2) .*gamma((k/2)));
            %             x        = linspace(0,max(bn),numel(bn));
            %             c        = chipdf(x,3);
            %             bar(bn,hn./trapz(bn,hn)), hold on
            %             plot(x,c./trapz(x,c),'r')
        end

    end

    methods (Access=public)

        function ExportTrajectoryDataToCsv(obj)
            % Export all trajectory data
            if obj.params.exportPDBtrajectories
                for tIdx = 1:numel(obj.allTraj)
                    t           = table();
                    N           = numel(obj.allTraj(tIdx).x(:));
                    t.ID        = ones(N,1).*obj.allTraj(tIdx).ID;
                    t.frame     = obj.allTraj(tIdx).frames(:);
                    t.x         = obj.allTraj(tIdx).x(:);
                    t.y         = obj.allTraj(tIdx).y(:);
                    t.z         = obj.allTraj(tIdx).z(:);
                    t.confinementScore = obj.allTraj(tIdx).confinementScore(:);
                    t.confined  = double(obj.allTraj(tIdx).confined(:));
                    t.alpha     = obj.allTraj(tIdx).alpha(:);
                    t.beta      = obj.allTraj(tIdx).beta(:);
                    t.diffusion = obj.allTraj(tIdx).diffusionConst';
                    t.Lc        = obj.allTraj(tIdx).Lc';
                    t.Kc        = obj.allTraj(tIdx).Kc';
                    t.driftMagnitude = sqrt(sum((obj.allTraj(tIdx).drift).^2,2));
                    t.originDataset  = repmat(obj.allTraj(tIdx).originDataset,[numel(t.Lc),1]);
                    t.datasetIdx     = repmat(obj.allTraj(tIdx).datasetIdx,[numel(t.Lc),1]);
                    %              writetable(t,fullfile(resultFolderName,['Traj_',num2str(tIdx),'.csv']));
                    writetable(t,fullfile(obj.params.allTrajectoryFolderName,['Traj_',num2str(obj.allTraj(tIdx).ID),'.csv']));
                end

                % export unconfined trajectories
                if numel(obj.unconfined)>1
                    for tIdx = 1:numel(obj.unconfined)
                        t           = table();
                        N           = numel(obj.unconfined(tIdx).x(:));
                        t.ID        = ones(N,1).*obj.unconfined(tIdx).ID;
                        t.frame     = obj.unconfined(tIdx).frames(:);
                        t.x         = obj.unconfined(tIdx).x(:);
                        t.y         = obj.unconfined(tIdx).y(:);
                        t.z         = obj.unconfined(tIdx).z(:);
                        t.confinementScore = obj.unconfined(tIdx).confinementScore(:);
                        t.confined  = double(obj.unconfined(tIdx).confined(:));
                        t.alpha     = ones(numel(t.x),1).*obj.unconfined(tIdx).alpha(:);
                        t.beta      = ones(numel(t.x),1).*obj.unconfined(tIdx).beta(:);
                        t.diffusion = ones(numel(t.x),1).*obj.unconfined(tIdx).diffusionConst;
                        t.Lc        = ones(numel(t.x),1).*obj.unconfined(tIdx).Lc;
                        t.Kc        = ones(numel(t.x),1).*obj.unconfined(tIdx).Kc;
                        t.driftMagnitude =  ones(numel(t.x),1).*norm(obj.unconfined(tIdx).drift);
                        t.originDataset  = repmat(obj.unconfined(tIdx).originDataset,[numel(t.Lc),1]);
                        t.datasetIdx     = repmat(obj.unconfined(tIdx).datasetIdx,[numel(t.Lc),1]);
                        writetable(t,fullfile(obj.params.unconfinedTrajectoryFolderName,['Traj_',num2str(tIdx),'.csv']));
                    end
                end

                % Export confined trajectories
                if numel(obj.confined)>1
                    for tIdx = 1:numel(obj.confined)
                        t           = table();
                        N           = numel(obj.confined(tIdx).x);
                        t.ID        = ones(N,1).*obj.confined(tIdx).ID;
                        t.frame     = obj.confined(tIdx).frames(:);
                        t.x         = obj.confined(tIdx).x(:);
                        t.y         = obj.confined(tIdx).y(:);
                        t.z         = obj.confined(tIdx).z(:);
                        t.confinementScore = obj.confined(tIdx).confinementScore(:);
                        t.confined  = double(obj.confined(tIdx).confined(:));
                        t.alpha     = ones(numel(t.x),1).*obj.confined(tIdx).alpha(:);
                        t.beta      = ones(numel(t.x),1).*obj.confined(tIdx).beta(:);
                        t.diffusion = ones(numel(t.x),1).*obj.confined(tIdx).diffusionConst;
                        t.Lc        = ones(numel(t.x),1).*obj.confined(tIdx).Lc;
                        t.Kc             = ones(numel(t.x),1).*obj.confined(tIdx).Kc;
                        t.driftMagnitude = ones(numel(t.x),1).*norm(obj.confined(tIdx).drift);
                        t.originDataset  = repmat(obj.confined(tIdx).originDataset,[numel(t.Lc),1]);
                        t.datasetIdx     = repmat(obj.confined(tIdx).datasetIdx,[numel(t.Lc),1]);
                        writetable(t,fullfile(obj.params.confinedTrajectoryFolderName,['Traj_',num2str(tIdx),'.csv']));
                    end
                end
            end

        end

        function ExportStatisticalParameters(obj)
            % Export to csv  statistical parameters used for classification
            % alpha,D,Kc,Lc, and the drift vector's magnitude.

            % -- Confined ---
            % diffusionConst
            diffConstConfined  = [obj.confined(:).diffusionConst];
            tc                 = table();
            tc.diffusionConst  = diffConstConfined';
            tc.originDataset   = {obj.confined(:).originDataset}';
            tc.datasetIdx      = [obj.confined(:).datasetIdx]';
            writetable(tc,fullfile(obj.params.resultFolderConfined,'diffusionConst.csv'))

            % alpha
            alphaConfined = [obj.confined(:).alpha];
            tc            = table();
            tc.alpha      = alphaConfined';
            tc.originDataset = {obj.confined(:).originDataset}';
            tc.datasetIdx    = [obj.confined(:).datasetIdx]';
            writetable(tc,fullfile(obj.params.resultFolderConfined,'alpha.csv'))

            % Lc
            lcConfined = [obj.confined(:).Lc];
            tc         = table();
            tc.Lc      = lcConfined';
            tc.originDataset = {obj.confined(:).originDataset}';
            tc.datasetIdx    = [obj.confined(:).datasetIdx]';
            writetable(tc,fullfile(obj.params.resultFolderConfined,'Lc.csv'))

            % drift vector magnitude
            driftConfined = zeros(1,numel(obj.confined));
            for cIdx =1:numel(obj.confined)
                driftConfined(cIdx) = norm(obj.confined(cIdx).drift);
            end
            tc                = table();
            tc.driftMagnitude = driftConfined';
            tc.originDataset  = {obj.confined(:).originDataset}';
            tc.datasetIdx     = [obj.confined(:).datasetIdx]';
            writetable(tc,fullfile(obj.params.resultFolderConfined,'driftMagnitude.csv'))

            % Kc
            kcConfined = [obj.confined(:).Kc];
            tc         = table();
            tc.Kc      = kcConfined';
            tc.originDataset = {obj.confined(:).originDataset}';
            tc.datasetIdx    = [obj.confined(:).datasetIdx]';
            writetable(tc,fullfile(obj.params.resultFolderConfined,'Kc.csv'))

            %--- Unconfined -----
            % diffusionConst
            diffConstUnconfined  = [obj.unconfined(:).diffusionConst];
            tc                   = table();
            tc.diffusionConst    = diffConstUnconfined';
            tc.originDataset     = {obj.unconfined(:).originDataset}';
            tc.datasetIdx        = [obj.unconfined(:).datasetIdx]';
            writetable(tc,fullfile(obj.params.resultFolderUnconfined,'diffusionConst.csv'))

            % alpha
            alphaUnconfined  = [obj.unconfined(:).alpha];
            tc               = table();
            tc.alpha         = alphaUnconfined';
            tc.originDataset = {obj.unconfined(:).originDataset}';
            tc.datasetIdx    = [obj.unconfined(:).datasetIdx]';
            writetable(tc,fullfile(obj.params.resultFolderUnconfined,'alpha.csv'))

            % Lc
            lcUnconfined = [obj.unconfined(:).Lc];
            tc           = table();
            tc.Lc        = lcUnconfined';
            tc.originDataset = {obj.unconfined(:).originDataset}';
            tc.datasetIdx    = [obj.unconfined(:).datasetIdx]';
            writetable(tc,fullfile(obj.params.resultFolderUnconfined,'Lc.csv'))

            % drift vector magnitude
            driftUnconfined = zeros(1,numel(obj.unconfined));
            for cIdx =1:numel(obj.unconfined)
                driftUnconfined(cIdx) = norm(obj.unconfined(cIdx).drift);
            end
            tc                = table();
            tc.driftMagnitude = driftUnconfined';
            tc.originDataset  = {obj.unconfined(:).originDataset}';
            tc.datasetIdx     = [obj.unconfined(:).datasetIdx]';
            writetable(tc,fullfile(obj.params.resultFolderUnconfined,'driftMagnitude.csv'))

            % Kc
            kcUnconfined = [obj.unconfined(:).Kc];
            tc           = table();
            tc.Kc        = kcUnconfined';
            tc.originDataset = {obj.unconfined(:).originDataset}';
            tc.datasetIdx    = [obj.unconfined(:).datasetIdx]';
            writetable(tc,fullfile(obj.params.resultFolderUnconfined,'Kc.csv'))

            % -- All trajectories ---

            originDB           = [];
            dsIdx              = [];
            for tIdx =1:length(obj.allTraj)
                originDB = [originDB; repmat(obj.allTraj(tIdx).originDataset,[numel(obj.allTraj(tIdx).diffusionConst),1])];
                dsIdx    = [dsIdx;repmat(obj.allTraj(tIdx).datasetIdx,[numel(obj.allTraj(tIdx).diffusionConst),1])];
            end

            % diffusionConst
            tc                 = table();
            tc.diffusionConst  = [obj.allTraj(:).diffusionConst]';
            tc.originDataset   = originDB;
            tc.datasetIdx      = dsIdx;
            writetable(tc,fullfile(obj.params.resultFolderAll,'diffusionConst.csv'))

            % alpha
            tc               = table();
            tc.alpha         = [obj.allTraj(:).alpha]';
            tc.originDataset = originDB;
            tc.datasetIdx    = dsIdx;
            writetable(tc,fullfile(obj.params.resultFolderAll,'alpha.csv'))

            % Lc
            tc               = table();
            tc.Lc            = [obj.allTraj(:).Lc]';
            tc.originDataset = originDB;
            tc.datasetIdx    = dsIdx;
            writetable(tc,fullfile(obj.params.resultFolderAll,'Lc.csv'))

            % Drift vector magnitude
            tc                = table();
            tc.driftMagnitude = [obj.allTraj(:).driftMagnitude]';
            tc.originDataset  = originDB;
            tc.datasetIdx     = dsIdx;
            writetable(tc,fullfile(obj.params.resultFolderAll,'driftMagnitude.csv'))

            % Kc
            tc               = table();
            tc.Kc            = [obj.allTraj(:).Kc]';
            tc.originDataset = originDB;
            tc.datasetIdx    = dsIdx;
            writetable(tc,fullfile(obj.params.resultFolderAll,'Kc.csv'))


            % C vs U collect stats (Yuze 20250129)
            t                   = table();
            t.datasetIdx        = unique(dsIdx);

            nDatasets = length(t.datasetIdx);

            oriDataset = strings(nDatasets, 1);
            allTrajCount = zeros(nDatasets, 1);
            allContrib = zeros(nDatasets, 1);
            confineCount = zeros(nDatasets, 1);
            confContrib = zeros(nDatasets, 1);
            unconfineCount = zeros(nDatasets, 1);
            unconfContrib = zeros(nDatasets, 1);
            confPerc = zeros(nDatasets, 1);
            unconfPerc = zeros(nDatasets, 1);

            for i = 1 : nDatasets
                oriDataset(i) = obj.allTraj(find([obj.allTraj.datasetIdx] == t.datasetIdx(i), 1)).originDataset;
                allTrajCount(i) = sum([obj.allTraj.datasetIdx] == t.datasetIdx(i));
                allContrib(i) = allTrajCount(i) / numel(obj.allTraj) * 100;
                confineCount(i) = sum([obj.confined.datasetIdx] == t.datasetIdx(i));
                confContrib(i) = confineCount(i) / numel(obj.confined) * 100;
                unconfineCount(i) = sum([obj.unconfined.datasetIdx] == t.datasetIdx(i));
                unconfContrib(i) = unconfineCount(i) / numel(obj.unconfined) * 100;
                confPerc(i) = confineCount(i) / (confineCount(i) + unconfineCount(i)) * 100;
                unconfPerc(i) = unconfineCount(i) / (confineCount(i) + unconfineCount(i)) * 100;
            end

            t.allTrajCount      = allTrajCount;
            t.allContrib        = allContrib;
            t.confineCount      = confineCount;
            t.confContrib       = confContrib;
            t.confContrib       = confContrib;
            t.unconfineCount    = unconfineCount;
            t.unconfContrib     = unconfContrib;
            t.confPerc          = confPerc;
            t.unconfPerc        = unconfPerc;
            t.originDataset     = oriDataset;
            writetable(t,fullfile(obj.params.resultFolderAll,'PerFOVstats.csv'))
        end

        function ExportTrajectoriesAsPDB(obj)
            % export trajectories in pdb format
            if obj.params.exportPDBtrajectories
                %             pdbPath = fullfile(obj.params.resultFolder,obj.params.datasetName,'PDB');
                %             [~,~]   = mkdir(pdbPath);
                for tIdx = 1:numel(obj.allTraj)
                    trajConf    = struct();
                    pdbFileName = fullfile(obj.params.pdbPath, ['traj_',num2str(tIdx),'.pdb']);
                    trajConf.X          = obj.allTraj(tIdx).x(:);
                    trajConf.Y          = obj.allTraj(tIdx).y(:);
                    trajConf.Z          = obj.allTraj(tIdx).z(:);
                    trajConf.outFile    = pdbFileName;
                    trajConf.recordName = repmat({'ATOM'},numel(trajConf.X),1);
                    trajConf.atomNum    = (1:numel(trajConf.X))';
                    for cIdx = 1:numel(trajConf.X)
                        if obj.allTraj(tIdx).confined(cIdx)
                            trajConf.atomName{cIdx}   = 'N';
                            trajConf.element{cIdx}    = 'N';
                        else
                            trajConf.atomName{cIdx}   = 'B';
                            trajConf.element{cIdx}    = 'B';
                        end
                    end
                    trajConf.altLoc     = [];
                    trajConf.resName    = repmat({'HIS'},numel(trajConf.X),1);
                    trajConf.chainID    = ones(numel(trajConf.X),1);
                    trajConf.resNum     = ones(numel(trajConf.X),1);
                    trajConf.occupancy  = ones(numel(trajConf.X),1);
                    trajConf.betaFactor = zeros(numel(trajConf.X),1);
                    trajConf.charge     = 1;
                    for cIdx = 1:numel(trajConf.X)-1
                        trajConf.connectedMonomers(cIdx,:) = [cIdx, cIdx+1];
                    end
                    obj.Trajectory2PDB(trajConf);
                end
            end
        end

        function ExportAssociationDissociationData(obj)

            % Export Association historgrams to CSV
            %          resultFolderName = fullfile(obj.params.resultFolder,obj.params.datasetName,'Association');
            %          [~,~]            = mkdir(resultFolderName);

            % export association times
            t                = table();
            t.values         = obj.associationTime.values(:);
            writetable(t,fullfile(obj.params.associationFolderName,'AssociationTimes.csv'));

            % export association histogram
            t                = table();
            t.bins           = obj.associationTime.bins(:);
            t.prob           = [obj.associationTime.histogram(:); 0]; % To match number of bins for table output
            writetable(t,fullfile(obj.params.associationFolderName,'Histogram.csv'));

            % export association fit
            if ~isempty(obj.associationTime.fit)
                t   = table();
                cn  = coeffnames(obj.associationTime.fit);  % names
                cv  = coeffvalues(obj.associationTime.fit); % values

                if obj.params.exportCI
                    CI  = confint(obj.associationTime.fit);
                    cv  = [cv; CI];
                end

                for tIdx = 1:numel(cn); t.(cn{tIdx}) = cv(:, tIdx); end
                % add the R square
                if obj.params.exportCI
                    t.rsquare = [obj.associationTime.fitStats.rsquare; nan; nan];
                    t.BIC = [obj.associationTime.fitStats.BIC; nan; nan];
                else
                    t.rsquare = obj.associationTime.fitStats.rsquare;
                    t.BIC = obj.associationTime.fitStats.BIC;
                end

                % export
                writetable(t,fullfile(obj.params.associationFolderName,'HistogramFit_best.csv'));
            end


            if ~isempty(obj.associationTime.allModels)
                nModel = numel(obj.associationTime.allModels);
                t   = table();
                cn  = coeffnames(obj.associationTime.allModels{end});  % names
                cv  = nan(nModel, numel(cn)); % values

                rsquares = zeros(nModel, 1);
                BICs = zeros(nModel, 1);

                for mIdx = 1:nModel
                    rsquares(mIdx) = obj.associationTime.allGOF{mIdx}.rsquare;
                    BICs(mIdx) = obj.associationTime.allGOF{mIdx}.BIC;

                    if mIdx == nModel
                        cv(mIdx, :) = coeffvalues(obj.associationTime.allModels{end});
                    else
                        cur_cv = coeffvalues(obj.associationTime.allModels{mIdx});
                        num_cur_cv = numel(cur_cv);
                        cv(mIdx, 1:num_cur_cv/2) = cur_cv(1:num_cur_cv/2);
                        cv(mIdx, numel(cn)/2+1:numel(cn)/2+num_cur_cv/2) = cur_cv(num_cur_cv/2+1:end);
                    end
                end

                for tIdx = 1:numel(cn)
                    t.(cn{tIdx}) = cv(:,tIdx);
                end
                % add the R square
                t.rsquare = rsquares;
                t.BIC = BICs;
                % export
                writetable(t,fullfile(obj.params.associationFolderName,'HistogramFit_all.csv'));
            end

            % Export Dissociation historgrams to CSV
            %           resultFolderName = fullfile(obj.params.resultFolder,obj.params.datasetName,'Dissociation');
            %          [~,~]             = mkdir(resultFolderName);

            % Export dissociation times
            t                 = table();
            t.values          = obj.dissociationTime.values(:);
            writetable(t,fullfile(obj.params.dissociationFolderName,'DissociationTimes.csv'));

            % Export dissociation histogram
            t                 = table();
            t.bins            = obj.dissociationTime.bins(:);
            t.prob            = [obj.dissociationTime.histogram(:); 0]; % To match number of bins for table output
            writetable(t,fullfile(obj.params.dissociationFolderName,'Histogram.csv'));

            % export dissociation fit
            if ~isempty(obj.dissociationTime.fit)
                t   = table();
                cn  = coeffnames(obj.dissociationTime.fit);  % names
                cv  = coeffvalues(obj.dissociationTime.fit); % values

                if obj.params.exportCI
                    CI  = confint(obj.dissociationTime.fit);
                    cv  = [cv; CI];
                end

                for tIdx = 1:numel(cn); t.(cn{tIdx}) = cv(:, tIdx); end

                % add the R square
                if obj.params.exportCI
                    t.rsquare = [obj.dissociationTime.fitStats.rsquare; nan; nan];
                    t.BIC = [obj.dissociationTime.fitStats.BIC; nan; nan];
                else
                    t.rsquare = obj.dissociationTime.fitStats.rsquare;
                    t.BIC = obj.dissociationTime.fitStats.BIC;
                end

                % export
                writetable(t,fullfile(obj.params.dissociationFolderName,'HistogramFit_best.csv'));
            end


            if ~isempty(obj.dissociationTime.allModels)
                nModel = numel(obj.dissociationTime.allModels);
                t   = table();
                cn  = coeffnames(obj.dissociationTime.allModels{end});  % names
                cv  = nan(nModel, numel(cn)); % values

                rsquares = zeros(nModel, 1);
                BICs = zeros(nModel, 1);

                for mIdx = 1:nModel
                    rsquares(mIdx) = obj.dissociationTime.allGOF{mIdx}.rsquare;
                    BICs(mIdx) = obj.dissociationTime.allGOF{mIdx}.BIC;

                    if mIdx == nModel
                        cv(mIdx, :) = coeffvalues(obj.dissociationTime.allModels{end});
                    else
                        cur_cv = coeffvalues(obj.dissociationTime.allModels{mIdx});
                        num_cur_cv = numel(cur_cv);
                        cv(mIdx, 1:num_cur_cv/2) = cur_cv(1:num_cur_cv/2);
                        cv(mIdx, numel(cn)/2+1:numel(cn)/2+num_cur_cv/2) = cur_cv(num_cur_cv/2+1:end);
                    end
                end

                for tIdx = 1:numel(cn)
                    t.(cn{tIdx}) = cv(:,tIdx);
                end
                % add the R square
                t.rsquare = rsquares;
                t.BIC = BICs;
                % export
                writetable(t,fullfile(obj.params.dissociationFolderName,'HistogramFit_all.csv'));
            end

        end

        function MouseOverFunction(obj,eventSource,varargin)%UNFINISHED
            % function to apply in the event of mouse motion over figure
            figHandle = eventSource;
            mousePos  = get(figHandle,'CurrentPoint');
            % the axes are normalized to [0 1]

            %             disp(mousePos);

        end

        function ParseInputParams(obj,paramsIn)
            % params should appear in name value pairs
            % e.g 'dt',0.02'
            numParamsIn = numel(paramsIn);
            assert(mod(numParamsIn,2)==0,'Input paremeters should appear in name-value pairs')
            for nIdx = 1:numParamsIn/2
                if isfield(obj.params,paramsIn{2*nIdx-1})
                    obj.params.(paramsIn{2*nIdx-1}) = paramsIn{2*nIdx};
                else
                    error(['The input parameter ' paramsIn{2*nIdx-1}, ' does not exist in the parameter list for this class.' ])
                end
            end

        end

        function CollectTrajectoryData(obj)
            % Collect trajectory data
            varnames = obj.data.Properties.VariableNames;
            % csv table structure,
            % col1 = track id
            % col2 = frame number
            % col 3-5: x,y,z coordinates

            obj.allTraj = obj.NewTrajectoryDataStruct;  % initialization
            next        = 1; % running trajectory index

            % Maximal axes value for trajectories' inspection boxes (Plotting)
            maxX     = 0;
            minX     = 0;
            maxY     = 0;
            minY     = 0;
            maxZ     = 0;
            minZ     = 0;

            for tIdx =1:max(obj.data.(varnames{1})) % for all trajectories

                trajInds      = find(obj.data.(varnames{1})==tIdx);
                frames        = obj.data.(varnames{2})(trajInds);
                framesSkipped = true; % indicator for frame gaps
                if all(diff(frames))==1; framesSkipped = false; end          % check for frame gaps
                if obj.params.allowFrameSkipping; framesSkipped = false; end % overide if allowed in parameters
                if (numel(trajInds)>obj.params.minNumPoints) && (framesSkipped==false)

                    obj.allTraj(next)                = obj.NewTrajectoryDataStruct;    % initialize data
                    obj.allTraj(next).ID             = tIdx; %next;
                    obj.allTraj(next).numFrames      = numel(trajInds);                 % in frames
                    obj.allTraj(next).frames         = obj.data.(varnames{2})(trajInds);
                    obj.allTraj(next).duration       = numel(trajInds)*obj.params.dt;    % in sec
                    obj.allTraj(next).x              = obj.data.(varnames{3})(trajInds) / 1000; % pos (µm)
                    obj.allTraj(next).y              = obj.data.(varnames{4})(trajInds) / 1000; % pos (µm)
                    obj.allTraj(next).z              = obj.data.(varnames{5})(trajInds) / 1000; % pos (µm)
                    obj.allTraj(next).center         = mean([obj.allTraj(next).x, obj.allTraj(next).y,obj.allTraj(next).z]);
                    % preallocations
                    obj.allTraj(next).alpha          = zeros(1,(obj.allTraj(next).numFrames-obj.params.numMSDpoints));
                    obj.allTraj(next).beta           = zeros(1,(obj.allTraj(next).numFrames-obj.params.numMSDpoints));
                    obj.allTraj(next).diffusionConst = zeros(1,obj.allTraj(next).numFrames);
                    obj.allTraj(next).Lc             = zeros(1,obj.allTraj(next).numFrames);
                    obj.allTraj(next).Kc             = zeros(1,obj.allTraj(next).numFrames);
                    obj.allTraj(next).drift          = zeros(obj.allTraj(next).numFrames,3);
                    obj.allTraj(next).driftMagnitude = zeros(1,obj.allTraj(next).numFrames);
                    obj.allTraj(next).displacement   = obj.ComputeDisplacement(obj.allTraj(next).x,...
                        obj.allTraj(next).y,...
                        obj.allTraj(next).z);
                    % Fit a spline to the trajectory
                    obj.allTraj(next).spline         = obj.FitSplineToPath(obj.allTraj(next).x,...
                        obj.allTraj(next).y,...
                        obj.allTraj(next).z);
                    % compute angles between successive steps
                    obj.allTraj(next).angles         =obj.AnglesAlongTrajectory([obj.allTraj(next).x,...
                        obj.allTraj(next).y,...
                        obj.allTraj(next).z]);
                    if iscell(obj.params.datasetName)==false
                        obj.allTraj(next).originDataset  = obj.params.datasetName;
                        obj.allTraj(next).datasetIdx     = 1;
                    else
                        obj.allTraj(next).originDataset = obj.params.datasetName{obj.datasetIdx(trajInds(1))};
                        obj.allTraj(next).datasetIdx    = obj.datasetIdx(trajInds(1));
                    end

                    % Change single-quoted string to double-quoted (20250204)
                    obj.allTraj(next).originDataset = string(obj.allTraj(next).originDataset);

                    
                    % per frame values
                    for mIdx=1:obj.allTraj(next).numFrames
                        % valid indices for sliding window
                        validInds           = max(1,mIdx-obj.params.numMSDpoints):min(obj.allTraj(next).numFrames,mIdx+obj.params.numMSDpoints);
                        numPoints           = numel(validInds);
                        traj                = obj.NewTrajectoryDataStruct;
                        traj.numFrames      = numPoints;
                        % compute Lc,Kc, diffusion const in a sliding window
                        traj.x              = obj.allTraj(next).x(validInds);
                        traj.y              = obj.allTraj(next).y(validInds);
                        traj.z              = obj.allTraj(next).z(validInds);
                        traj.center         = mean([traj.x traj.y traj.z]);
                        traj.diffusionConst = obj.ComputeDiffusionCoefficient(traj,obj.params.dt);

                        obj.allTraj(next).diffusionConst(mIdx) = traj.diffusionConst;
                        obj.allTraj(next).Lc(mIdx)             = obj.ComputeLc(traj);
                        obj.allTraj(next).Kc(mIdx)             = obj.ComputeKc(traj,obj.params.dt);
                        obj.allTraj(next).drift(mIdx,:)        = obj.ComputeDrift(traj,obj.params.dt);
                        obj.allTraj(next).driftMagnitude(mIdx) = sqrt(sum(obj.allTraj(next).drift(mIdx,:).^2,2));
                        % Compute the squared-displacement of the trajectory in a sliding window
                        meanSquareDisplacement = obj.ComputeMeanSquareDisplacement(traj)';
                        % Compute the slope of the squareDisplacement (alpha)
                        times                  = (0:numPoints-1).*obj.params.dt;
                        obj.allTraj(next).times{mIdx} = times;
                        [obj.allTraj(next).alpha(mIdx),obj.allTraj(next).beta(mIdx)]=...
                            obj.ComputeAlphaAndBetaMSD(times,meanSquareDisplacement);
                        obj.allTraj(next).msd{mIdx}   = meanSquareDisplacement;
                        % Add reference to the file trajectory is
                        % registered in

                    end

                    % Find boundaries for axes (arond trajectory center of mass
                    Mx = max(obj.allTraj(next).x-obj.allTraj(next).center(1));
                    mx = min(obj.allTraj(next).x-obj.allTraj(next).center(1));
                    My = max(obj.allTraj(next).y-obj.allTraj(next).center(2));
                    my = min(obj.allTraj(next).y-obj.allTraj(next).center(2));
                    Mz = max(obj.allTraj(next).z-obj.allTraj(next).center(3));
                    mz = min(obj.allTraj(next).z-obj.allTraj(next).center(3));

                    if Mx>maxX; maxX = Mx;end
                    if mx<minX; minX = mx;end
                    if My>maxY; maxY = My;end
                    if my<minY; minY = my;end
                    if Mz>maxZ; maxZ = Mz;end
                    if mz<minZ; minZ = mz;end

                    next = next+1;
                end
            end
            % set inspect boundaries
            obj.inspectAxesBoundaries.all = [minX maxX+eps; minY maxY+eps; minZ maxZ+eps];

            % Update the number of trajectories after filtering short ones
            obj.numTraj = numel(obj.allTraj);
        end

        function ClassifyTrajectories(obj)
            % Classify each position of a trajectory into confined or unconfined state

            % Collect statistics from all trajectories
            DT     = [obj.allTraj(:).diffusionConst]'; % all diffusion const
            KT     = [obj.allTraj(:).Kc]';    % all Kc vals
            alphT  = [obj.allTraj(:).alpha]'; % all alpha vals
            LT     = [obj.allTraj(:).Lc]';    % all Lc
            %             driftT = [];
            %             for trajIdx = 1:numel(obj.allTraj); driftT = [driftT;obj.allTraj(trajIdx).drift];end
            normDriftT  = [obj.allTraj(:).driftMagnitude]';%sqrt(sum(driftT.^2,2));

            % Standardize the data
            LT         = obj.Standardize(LT);
            DT         = obj.Standardize(DT);
            alphT      = obj.Standardize(alphT);
            KT         = obj.Standardize(KT);
            normDriftT = obj.Standardize(normDriftT);

            % all feature matrix
            %             rT        = [alphT normDriftT DT LT];
            rT = [];
            if obj.params.classifyUsingAlpha; rT = [rT,alphT]; end
            if obj.params.classifyUsingDriftNorm; rT = [rT, normDriftT]; end
            if obj.params.classifyUsingDiffusion; rT = [rT, DT]; end
            if obj.params.classifyUsingLc; rT=[rT,LT]; end
            if obj.params.classifyUsingSpringConst; rT = [rT, KT]; end
            if isempty(rT); error('At least one feature must be used for classification'); end

            % fit a Gaussian mixing distribution with numberOfClasses components
            opt            = statset('MaxIter',1500);
            obj.classifier = fitgmdist(rT,obj.params.numberOfClasses,'Replicates',20,'Options',opt);
            classTrace     = zeros(1,obj.params.numberOfClasses);
            for cIdx =1:obj.params.numberOfClasses
                classTrace(cIdx) = trace(obj.classifier.Sigma(:,:,cIdx));
            end
            [~,Cinds] = min(classTrace);%sqrt(sum(f.mu.^2,2))); % confined charachterized by a smaller norm.
            confInds  = obj.classifier.cluster(rT)==Cinds;

            for tInd = 1:obj.numTraj
                fStart = sum([obj.allTraj(1:(tInd-1)).numFrames])+1;
                fEnd   = fStart+obj.allTraj(tInd).numFrames -1;
                posteriorProb = obj.classifier.posterior(rT(fStart:fEnd,:));
                obj.allTraj(tInd).confinementScore = posteriorProb(:,Cinds);
                obj.allTraj(tInd).confined         = confInds(fStart:fEnd);
            end

            obj.RemoveIsolatedClassPoints;

        end

        function RemoveIsolatedClassPoints(obj)
            % Remove isolated confined/unconfined points
            for tIdx = 1:obj.numTraj
                % confinedTemp = obj.allTraj(tIdx).confined;
                for pIdx = 2:numel(obj.allTraj(tIdx).confined)-1
                    % Go over the confined flag and remove singular points
                    cNeighbors = (obj.allTraj(tIdx).confined(pIdx-1) & obj.allTraj(tIdx).confined(pIdx+1)) |...
                        (~obj.allTraj(tIdx).confined(pIdx-1) & ~obj.allTraj(tIdx).confined(pIdx+1)); % check if the neighbors have similar value
                    if cNeighbors && obj.allTraj(tIdx).confined(pIdx) ~= obj.allTraj(tIdx).confined(pIdx-1)
                        obj.allTraj(tIdx).confined(pIdx) = obj.allTraj(tIdx).confined(pIdx-1);% give it the neighbor's value
                    end
                end
            end
        end

        function CollectAssociationDissociationData(obj)
            % Extract dissociation and association rates
            for tIdx = 1:obj.numTraj
                % Dissociation time
                bD = bwlabel(obj.allTraj(tIdx).confined);
                for bIdx =1:max(bD)%2:2:max(bD)
                    % A confined segment must be between two
                    % unconfined segments (i.e. even index)
                    nb = find(bD==bIdx);
                    if numel(nb)>obj.params.minNumPoints
                        bD(nb)=(1:numel(nb)).*obj.params.dt;
                        obj.allTraj(tIdx).dissociationTime(end+1)= bD(nb(end));
                    end
                end

                % Association Time
                aD = bwlabel(~obj.allTraj(tIdx).confined);
                for bIdx = 1:max(aD)% 2:2:max(aD)
                    % An unconfined segment must be betwee two confined
                    % segments (i.e even index)
                    nb = find(aD==bIdx);
                    if numel(nb)>obj.params.minNumPoints
                        aD(nb) = (1:numel(nb)).*obj.params.dt;
                        obj.allTraj(tIdx).associationTime(end+1)=aD(nb(end));
                    end
                end
            end
        end

        function ConstructAssociationDissociationHist(obj)
            % Collect all association times and dissociation time
            aTime                        = [obj.allTraj(:).associationTime];
            dTime                        = [obj.allTraj(:).dissociationTime];
            % [aHist,binsAssociationHist]  = hist(aTime,floor(0.5*numel(aTime)));% take 10 % of the total number of points
            % [dHist,binsDissociationHist] = hist(dTime,floor(0.5*numel(dTime)));
            % aHist                        = aHist./sum(aHist);
            % dHist                        = dHist./sum(dHist);

            % fit association dissociation histograms
            [aHist, binsAssociationHist, associationFit, associationFitStats, assoAllModels, assoAllGOF] = obj.FitAssociationHistogram(aTime);
            [dHist, binsDissociationHist, dissociationFit, dissociationFitStats, dissoAllModels, dissoAllGOF] = obj.FitDissociationHistogram(dTime);


            % Save values
            obj.associationTime.histogram  = aHist;
            obj.associationTime.bins       = binsAssociationHist;
            obj.associationTime.values     = aTime;
            obj.associationTime.fit        = associationFit;  % Best fit
            obj.associationTime.fitStats   = associationFitStats;
            obj.associationTime.allModels  = assoAllModels;  % All fits
            obj.associationTime.allGOF     = assoAllGOF;
            obj.associationTime.mean       = mean(aTime);

            obj.dissociationTime.histogram = dHist;
            obj.dissociationTime.bins      = binsDissociationHist;
            obj.dissociationTime.values    = dTime;
            obj.dissociationTime.fit       = dissociationFit;  % Best fit
            obj.dissociationTime.fitStats  = dissociationFitStats;
            obj.dissociationTime.allModels  = dissoAllModels;  % All fits
            obj.dissociationTime.allGOF     = dissoAllGOF;
            obj.dissociationTime.mean      = mean(dTime);

            % Plot
            % obj.PlotAssociationDissociationHistograms
            % Plot done in obj.FitAssociationHistogram and obj.FitDissociationHistogram

        end

        function ClassifyConfinedUnconfinedTrajectories(obj)
            % Classify the trjectories in confined and non confined states
            minTrajLength   = obj.params.minNumPointsLongTraj; % minimal number of frames
            nextLong        = 1;
            nextConfined    = 1;
            obj.unconfined  = obj.NewTrajectoryDataStruct;  % initialized unconfined
            obj.confined    = obj.NewTrajectoryDataStruct;  % initialize confined

            minX = 0; mincX = 0;
            maxX = 0; maxcX = 0;
            minY = 0; mincY = 0;
            maxY = 0; maxcY = 0;
            minZ = 0; mincZ = 0;
            maxZ = 0; maxcZ = 0;

            % The model for fitting the main axis of the non-confined trajectories
            mainAxisModel = fittype(@(a,b,x,y)a*x+b*y,'independent',{'x','y'},'dependent','z');

            for tIdx = 1:obj.numTraj
                bLong     = bwlabel(~obj.allTraj(tIdx).confined); % all non confined time points
                bConfined = bwlabel(obj.allTraj(tIdx).confined);  % all confined time points

                for mIdx =1:max(bLong)
                    trajInds = find(bLong==mIdx);
                    if numel(trajInds)>=minTrajLength
                        obj.unconfined(nextLong)           = obj.NewTrajectoryDataStruct;
                        obj.unconfined(nextLong).ID        = tIdx;
                        obj.unconfined(nextLong).x         = obj.allTraj(tIdx).x(trajInds);
                        obj.unconfined(nextLong).y         = obj.allTraj(tIdx).y(trajInds);
                        obj.unconfined(nextLong).z         = obj.allTraj(tIdx).z(trajInds);
                        obj.unconfined(nextLong).numFrames = numel(trajInds);
                        obj.unconfined(nextLong).duration  = numel(trajInds)*obj.params.dt;
                        obj.unconfined(nextLong).confined  = false(1,numel(trajInds));
                        obj.unconfined(nextLong).frames    = obj.allTraj(tIdx).frames(trajInds);
                        obj.unconfined(nextLong).times     = (0:numel(trajInds)-1).*obj.params.dt;
                        obj.unconfined(nextLong).confinementScore = obj.allTraj(tIdx).confinementScore(trajInds);
                        obj.unconfined(nextLong).originDataset    = obj.allTraj(tIdx).originDataset;
                        obj.unconfined(nextLong).datasetIdx       = obj.allTraj(tIdx).datasetIdx;

                        % Compute the MSD, alpha and beta
                        obj.unconfined(nextLong).msd            = obj.ComputeMeanSquareDisplacement(obj.unconfined(nextLong));
                        obj.unconfined(nextLong).msd            = obj.unconfined(nextLong).msd'; % remove zero point
                        [obj.unconfined(nextLong).alpha, obj.unconfined(nextLong).beta]=...
                            obj.ComputeAlphaAndBetaMSD(obj.unconfined(nextLong).times,obj.unconfined(nextLong).msd);
                        % center of mass of the trajectory
                        obj.unconfined(nextLong).center         = obj.ComputeCenter(obj.unconfined(nextLong));
                        % radius (width) of the trajectory
                        obj.unconfined(nextLong).radius         = obj.ComputeRadius(obj.unconfined(nextLong));
                        % diffusion const nm^2 /s
                        obj.unconfined(nextLong).diffusionConst = obj.ComputeDiffusionCoefficient(obj.unconfined(nextLong),obj.params.dt);
                        % drift [ vector (nm)]
                        obj.unconfined(nextLong).drift          = obj.ComputeDrift(obj.unconfined(nextLong),obj.params.dt);
                        % compute the spring const for confinement
                        obj.unconfined(nextLong).Kc             = obj.ComputeKc(obj.unconfined(nextLong),obj.params.dt);
                        % compute Lc the radius of confinement
                        obj.unconfined(nextLong).Lc             = obj.ComputeLc(obj.unconfined(nextLong));
                        % Fit a spline to the path
                        obj.unconfined(nextLong).spline         = obj.FitSplineToPath(obj.unconfined(nextLong).x ,...
                            obj.unconfined(nextLong).y,...
                            obj.unconfined(nextLong).z);

                        % compute angles between successive step of the path
                        obj.unconfined(nextLong).angles        = obj.AnglesAlongTrajectory([obj.unconfined(nextLong).x ,...
                            obj.unconfined(nextLong).y,...
                            obj.unconfined(nextLong).z]);
                        % compute successuve displacment
                        obj.unconfined(nextLong).displacement  = obj.ComputeDisplacement(obj.unconfined(nextLong).x ,...
                            obj.unconfined(nextLong).y,...
                            obj.unconfined(nextLong).z);
                        % Find boundaries for axes (arond trajectory center of mass
                        Mx = max(obj.unconfined(nextLong).x-obj.unconfined(nextLong).center(1));
                        mx = min(obj.unconfined(nextLong).x-obj.unconfined(nextLong).center(1));
                        My = max(obj.unconfined(nextLong).y-obj.unconfined(nextLong).center(2));
                        my = min(obj.unconfined(nextLong).y-obj.unconfined(nextLong).center(2));
                        Mz = max(obj.unconfined(nextLong).z-obj.unconfined(nextLong).center(3));
                        mz = min(obj.unconfined(nextLong).z-obj.unconfined(nextLong).center(3));

                        if Mx>maxX; maxX = Mx;end
                        if mx<minX; minX = mx;end
                        if My>maxY; maxY = My;end
                        if my<minY; minY = my;end
                        if Mz>maxZ; maxZ = Mz;end
                        if mz<minZ; minZ = mz;end

                        x = obj.unconfined(nextLong).x-obj.unconfined(nextLong).x(1);
                        y = obj.unconfined(nextLong).y-obj.unconfined(nextLong).y(1);
                        z = obj.unconfined(nextLong).z-obj.unconfined(nextLong).z(1);

                        % compute the main axis of motion by a linear regression
                        [fitv,~] = fit([x, y],z, mainAxisModel,'StartPoint',rand(1,2));
                        obj.unconfined(nextLong).mainAxis.a = fitv.a;
                        obj.unconfined(nextLong).mainAxis.b = fitv.b;
                        obj.unconfined(nextLong).mainAxis.x = x;
                        obj.unconfined(nextLong).mainAxis.y = y;
                        obj.unconfined(nextLong).mainAxis.z = z;

                        % advance the trajectory index counter
                        nextLong = nextLong+1;
                    end
                end

                % Confined trajectories
                for mIdx = 1:max(bConfined)
                    trajInds = find(bConfined==mIdx);
                    if numel(trajInds)>=minTrajLength
                        obj.confined(nextConfined)           = obj.NewTrajectoryDataStruct;
                        obj.confined(nextConfined).ID        = tIdx;
                        obj.confined(nextConfined).x         = obj.allTraj(tIdx).x(trajInds);
                        obj.confined(nextConfined).y         = obj.allTraj(tIdx).y(trajInds);
                        obj.confined(nextConfined).z         = obj.allTraj(tIdx).z(trajInds);
                        obj.confined(nextConfined).numFrames = numel(trajInds);
                        obj.confined(nextConfined).duration  = numel(trajInds)*obj.params.dt;
                        obj.confined(nextConfined).confined  = true(1,numel(trajInds));
                        obj.confined(nextConfined).frames    = obj.allTraj(tIdx).frames(trajInds);
                        obj.confined(nextConfined).times     = (0:numel(trajInds)-1).*obj.params.dt; % time points of the trajectory
                        obj.confined(nextConfined).confinementScore = obj.allTraj(tIdx).confinementScore(trajInds);
                        obj.confined(nextConfined).originDataset    = obj.allTraj(tIdx).originDataset;
                        obj.confined(nextConfined).datasetIdx           = obj.allTraj(tIdx).datasetIdx;


                        % Compute square displacement
                        obj.confined(nextConfined).msd       = obj.ComputeMeanSquareDisplacement(obj.confined(nextConfined));%
                        obj.confined(nextConfined).msd       = obj.confined(nextConfined).msd'; % remove zero point
                        % Compute the MSD, alpha and beta
                        [obj.confined(nextConfined).alpha,obj.confined(nextConfined).beta]=...
                            obj.ComputeAlphaAndBetaMSD(obj.confined(nextConfined).times,...
                            obj.confined(nextConfined).msd); % fit alpha and beta MSD

                        % Center of mass of confined time points
                        obj.confined(nextConfined).center = obj.ComputeCenter(obj.confined(nextConfined));

                        % Compute radius
                        obj.confined(nextConfined).radius =  obj.ComputeRadius(obj.confined(nextConfined));

                        % Apparent diffusion const.
                        obj.confined(nextConfined).diffusionConst = obj.ComputeDiffusionCoefficient(obj.confined(nextConfined),obj.params.dt);
                        % apparent drift vector
                        obj.confined(nextConfined).drift          = obj.ComputeDrift(obj.confined(nextConfined),obj.params.dt);
                        % compute the spring const for confinement
                        obj.confined(nextConfined).Kc             = obj.ComputeKc(obj.confined(nextConfined),obj.params.dt);
                        % compute Lc the radius of confinement
                        obj.confined(nextConfined).Lc             = obj.ComputeLc(obj.confined(nextConfined));
                        % fit a spline to the path
                        obj.confined(nextConfined).spline         = obj.FitSplineToPath(obj.confined(nextConfined).x,...
                            obj.confined(nextConfined).y,...
                            obj.confined(nextConfined).z);

                        obj.confined(nextConfined).angles        = obj.AnglesAlongTrajectory([obj.confined(nextConfined).x,...
                            obj.confined(nextConfined).y,...
                            obj.confined(nextConfined).z]);

                        obj.confined(nextConfined).displacement = obj.ComputeDisplacement(obj.confined(nextConfined).x,...
                            obj.confined(nextConfined).y,...
                            obj.confined(nextConfined).z);

                        % Find boundaries for axes (centered at trajectory's center of mass)
                        Mcx = max(obj.confined(nextConfined).x-obj.confined(nextConfined).center(1));
                        mcx = min(obj.confined(nextConfined).x-obj.confined(nextConfined).center(1));
                        Mcy = max(obj.confined(nextConfined).y-obj.confined(nextConfined).center(2));
                        mcy = min(obj.confined(nextConfined).y-obj.confined(nextConfined).center(2));
                        Mcz = max(obj.confined(nextConfined).z-obj.confined(nextConfined).center(3));
                        mcz = min(obj.confined(nextConfined).z-obj.confined(nextConfined).center(3));

                        if Mcx>maxcX; maxcX = Mcx;end
                        if mcx<mincX; mincX = mcx;end
                        if Mcy>maxcY; maxcY = Mcy;end
                        if mcy<mincY; mincY = mcy;end
                        if Mcz>maxcZ; maxcZ = Mcz;end
                        if mcz<mincZ; mincZ = mcz;end

                        nextConfined = nextConfined+1;
                    end
                end
            end
            obj.inspectAxesBoundaries.unconfined = [minX-eps maxX+eps; minY-eps maxY+eps; minZ-eps maxZ+eps];
            obj.inspectAxesBoundaries.confined   = [mincX-eps maxcX+eps; mincY-eps maxcY+eps; mincZ-eps maxcZ+eps];
        end
        % TODO: classify unconfined to directed and diffusive motion

        function PlotIndividualTrajectories(obj,tInd)

            if ~isempty(get(obj.handles.subplot1,'Children')); delete(get(obj.handles.subplot1,'Children')); end
            x = obj.allTraj(tInd).x-obj.allTraj(tInd).center(1);
            y = obj.allTraj(tInd).y-obj.allTraj(tInd).center(2);
            z = obj.allTraj(tInd).z-obj.allTraj(tInd).center(3);


            xConfined = x;
            %             xConfined(~obj.allTraj(tInd).confined)=NaN;
            yConfined = y;
            %             yConfined(~obj.allTraj(tInd).confined)=NaN;
            zConfined = z;
            %             zConfined(~obj.allTraj(tInd).confined)=NaN;

            xUnconfined = x; xUnconfined(obj.allTraj(tInd).confined)=NaN;
            yUnconfined = y; yUnconfined(obj.allTraj(tInd).confined)=NaN;
            zUnconfined = z; zUnconfined(obj.allTraj(tInd).confined)=NaN;

            line('XData',xConfined,...
                'YData',yConfined,...
                'ZData',zConfined,...
                'Color',obj.colors.confined,...
                'Marker','o',...
                'MarkerFaceColor',obj.colors.confined,...
                'MarkerEdgeColor',obj.colors.confined,...
                'LineWidth',2,...
                'Parent',obj.handles.subplot1);

            line('XData',xUnconfined,...
                'YData',yUnconfined,...
                'ZData',zUnconfined,...
                'Color',obj.colors.unconfined,...
                'Marker','o',...
                'MarkerFaceColor',obj.colors.unconfined,...
                'MarkerEdgeColor',obj.colors.unconfined,...
                'LineWidth',2,...
                'Parent',obj.handles.subplot1);


            % Add spline
            %             cp  = obj.FitSplineToPath(x,y,z);
            %             spc = obj.SplineCurvature(cp,obj.params.dt);
            %             f1  = fnplt(cp,'r');
            %             plot3(obj.handles.subplot1,f1(1,:),f1(2,:),f1(3,:),'r','LineWidth',2)
            set(obj.handles.subplot1,...
                'LineWidth',2,...
                'FontSize',20,...
                'FontName','Arial','Color','none');

            xlabel(obj.handles.subplot1,'X (nm)');
            ylabel(obj.handles.subplot1,'Y (nm)');
            zlabel(obj.handles.subplot1,'Z (nm)');

            % add time point annotation
            for tIdx =[1,numel(obj.allTraj(tInd).x)]
                text(x(tIdx),...
                    y(tIdx),...
                    z(tIdx),...
                    num2str(tIdx),'FontSize',10);
            end

            title(obj.handles.subplot1,['trajectory ' num2str(tInd)],'FontSize',24)
            daspect([1 1 1])
            view(obj.handles.subplot1,[-37.5 30])
            set(obj.handles.subplot1,'NextPlot','ReplaceChildren');


            % plot the confinement score and classification
            plot(obj.handles.subplot2,1:numel(obj.allTraj(tInd).confinementScore),obj.allTraj(tInd).confinementScore,...
                'LineWidth',3), hold on
            set(obj.handles.subplot2,'NextPlot','Add');
            plot(obj.handles.subplot2,find(obj.allTraj(tInd).confined),...
                obj.allTraj(tInd).confinementScore(obj.allTraj(tInd).confined),'o',...
                'MarkerFaceColor',obj.colors.confined,...
                'MarkerEdgeColor',obj.colors.confined)
            plot(obj.handles.subplot2,find(~obj.allTraj(tInd).confined),...
                obj.allTraj(tInd).confinementScore(~obj.allTraj(tInd).confined),'o',...
                'MarkerFaceColor',obj.colors.unconfined,...
                'MarkerEdgeColor',obj.colors.unconfined)

            set(obj.handles.subplot2,...
                'FontSize',24,...
                'LineWidth',3,...
                'FontName','Arial',...
                'XLim',[1 numel(obj.allTraj(tInd).confined)],...
                'YLim',[-0.05 1.1])
            xlabel(obj.handles.subplot2,'Time (sec)')
            ylabel(obj.handles.subplot2,'Confinement score')
            cameratoolbar
            set(obj.handles.subplot2,'NextPlot','Replace')
        end

        function PlotIndividualUnconfinedTrajectories(obj,tInd)
            cla
            x = obj.unconfined(tInd).x-obj.unconfined(tInd).center(1);
            y = obj.unconfined(tInd).y-obj.unconfined(tInd).center(2);
            z = obj.unconfined(tInd).z-obj.unconfined(tInd).center(3);
            line('XData',x,...
                'YData',y,...
                'ZData',z,...
                'Parent',obj.handles.unconfinedAx,...
                'LineWidth',2,...
                'Marker','o',...
                'MarkerFaceColor',obj.colors.unconfined,...
                'MarkerEdgeColor',obj.colors.unconfined);
            set(obj.handles.unconfinedAx,'NextPlot','Add');

            set(obj.handles.unconfinedAx,'LineWidth',2,...
                'FontSize',20,...
                'Color','none',...
                'FontName','Arial');

            xlabel(obj.handles.unconfinedAx,'X (nm)')
            ylabel(obj.handles.unconfinedAx,'Y (nm)')
            zlabel(obj.handles.unconfinedAx,'Z (nm)')

            % add time point annotation
            for tIdx =1:numel(obj.unconfined(tInd).x)
                text(x(tIdx),...
                    y(tIdx),...
                    z(tIdx),...
                    num2str(tIdx),'FontSize',10);
            end

            % add spline
            %             fnplt(obj.FitSplineToPath(x,y,z),'r');

            title(obj.handles.unconfinedAx,['unconfined trajectory ' num2str(tInd)],'FontSize',24)

            set(obj.handles.unconfinedAx,'NextPlot','ReplaceChildren','Box','on');

        end

        function PlotIndividualConfinedTrajectories(obj,tInd)
            cla
            x = obj.confined(tInd).x-obj.confined(tInd).center(1);
            y = obj.confined(tInd).y-obj.confined(tInd).center(2);
            z = obj.confined(tInd).z-obj.confined(tInd).center(3);
            line('XData',x,...
                'YData',y,...
                'ZData',z,...
                'Parent',obj.handles.confinedAx,...
                'LineWidth',2,...
                'Marker','o',...
                'MarkerFaceColor',obj.colors.confined,...
                'MarkerEdgeColor',obj.colors.confined);
            set(obj.handles.confinedAx,'LineWidth',2,...
                'FontSize',20,...
                'FontName','Arial',...
                'Color','none',...
                'NextPlot','Add');

            xlabel(obj.handles.confinedAx,'X (nm)'),
            ylabel(obj.handles.confinedAx,'Y (nm)')
            zlabel(obj.handles.confinedAx,'Z (nm)')

            % add time point annotation
            for tIdx =1:numel(obj.confined(tInd).x)
                text(x(tIdx),...
                    y(tIdx),...
                    z(tIdx),...
                    num2str(tIdx),'FontSize',10);
            end
            %             fnplt(obj.FitSplineToPath(x,y,z),'r');
            title(obj.handles.confinedAx,['Confined trajectory ' num2str(tInd)],'FontSize',24)
            set(obj.handles.confinedAx,'NextPlot','ReplaceChildren','Box','on');

        end

        function TrajSliderMovement(obj,sliderHandle,varargin)
            % Slider callback
            obj.PlotIndividualTrajectories(floor(get(sliderHandle,'Value')));
        end

        function UnconfinedTrajSliderMovement(obj,sliderHandle,varargin)
            % Slider callback
            obj.PlotIndividualUnconfinedTrajectories(floor(get(sliderHandle,'Value')));
        end

        function ConfinedTrajSliderMovement(obj,sliderHandle,varargin)
            % Slider callback
            obj.PlotIndividualConfinedTrajectories(floor(get(sliderHandle,'Value')));
        end

        function ConstructUnconfinedLengthHist(obj)
            % the distribution of long trajectories (unconfined)
            unconfinedTimes = [obj.unconfined.duration];
            figure('Units','norm')
            % histogram with 10% of the total number of traj
            [lHist,lBins] = hist(unconfinedTimes,0.1*floor(numel(unconfinedTimes)));
            bar(lBins,lHist)
            xlabel('Trajectory during (s)')
            ylabel('Frequency');
            title('Non-confined trajectories')
            set(gca,'FontSize',24,'FontName','Arial')

        end

        function [y, edges, fitModel, fitStats, fit_models, gof] = FitAssociationHistogram(obj, association_times)


            % Integrated Stan's codes --- Start --- (Yuze 20250128)
            edges = min(association_times)-obj.params.dt/2 : obj.params.dt : max(association_times)+obj.params.dt/2;
            [y, edges] = histcounts(association_times, edges, 'Normalization','pdf');






            % take the midpoint
            x = (edges(1:end-1)+edges(2:end))./2;


            % calculte one exponential fitting
            f_1 = @(A,tau_1,x)(A.*exp(-x./tau_1));
            f_2 = @(A,f1,tau_1,tau_2,x)(A.*(f1.*exp(-x./tau_1) + (1-f1).*exp(-x./tau_2)));
            f_3 = @(A,f1,f2,tau_1,tau_2,tau_3,x)(A.*(f1.*exp(-x./tau_1) + f2.*exp(-x./tau_2) + (1-f1-f2).*exp(-x./tau_3)));
            f_4 = @(A,f1,f2,f3,tau_1,tau_2,tau_3,tau_4,x)(A.*(f1.*exp(-x./tau_1) + f2.*exp(-x./tau_2) + f3.*exp(-x./tau_3) + (1-f1-f2-f3).*exp(-x./tau_4)));
            f_5 = @(A,f1,f2,f3,f4,tau_1,tau_2,tau_3,tau_4,tau_5,x)(A.*(f1.*exp(-x./tau_1) + f2.*exp(-x./tau_2) + f3.*exp(-x./tau_3) + f4.*exp(-x./tau_4) + (1-f1-f2-f3-f4).*exp(-x./tau_5)));

            f = {f_1, f_2, f_3, f_4, f_5};

            n_fit = min(5, floor(length(x)/2));
            fit_models = cell(n_fit,1);
            gof = cell(n_fit,1);
            bic = zeros(1,n_fit);

            for i = 1 : n_fit

                fo = fitoptions('Method','NonlinearLeastSquares',...
                    'Upper', [Inf ones(1,i-1) Inf(1,i)],...
                    'Lower', zeros(1,2*i),...
                    'StartPoint', ones(1,2*i),...
                    'Robust','LAR');

                [fit_models{i}, gof{i}] = fit(x', y', f{i}, fo);

                bic(i) = obj.bic_calculator(y, fit_models{i}(x)', 1);

                gof{i}.BIC = bic(i);
            end

            [~, best] = min(bic);

            fitModel = fit_models{best};
            fitStats = gof{best};

            % Plot Data and fits
            fig = figure("Name", "Association Histogram");
            clf
            hold on

            x_grid = linspace(x(1),x(end),100);

            for i = 1 : best
                plot(x_grid,fit_models{i}(x_grid)','LineWidth',5, ...
                    "DisplayName", sprintf('#Pop=%d (BIC=%.2f)', i, bic(i)))
                if best == 1 && n_fit > 1
                    plot(x_grid,fit_models{2}(x_grid)',':','LineWidth',5,'Color',[0.5020 0.5020 0], ...
                        "DisplayName", sprintf('#Pop=2 (BIC=%.2f)', bic(2)))
                end
            end

            scatter(x,y,50,'k','Filled', "DisplayName", 'Empirical')

            legend("Location", "best")

            title('Association times')
            xlabel('Time (seconds)')
            ylabel('pdf')

            axis([0 max(x) 0 max(y)])
            set(gca,'FontName','Arial','FontSize',12,'FontWeight','bold')
            shg


            if obj.params.exportFigures
                savefig(fig,fullfile(obj.params.figureFolderName,'AssociationHistogram.fig'))
            end


            fig = figure("Name", "Association Fit BIC");
            clf
            hold on
            plot(1:n_fit, bic, 'LineWidth',5, 'Color', [0 0 0])
            scatter(best, bic(best), 100, 'red', 'filled')
            title('Association fit BIC curve')
            xlabel('Number of exponentials')
            ylabel('BIC')
            set(gca, 'XTick', 1:n_fit)
            set(gca,'FontName','Arial','FontSize',12,'FontWeight','bold')
            shg

            if obj.params.exportFigures
                savefig(fig,fullfile(obj.params.figureFolderName,'AssociationBIC.fig'))
            end

            % Integrated Stan's codes --- End ---

            if length(y) < 3  % In case #observations<=#coefficients no CI
                disp("Warning: Data points too few for fitting -- Not fitting.")
                fitModel = [];
                return
            end

        end



        function [y, edges, fitModel, fitStats, fit_models, gof] = FitDissociationHistogram(obj, dissociation_times)


            % Integrated Stan's codes --- Start ---
            edges = min(dissociation_times)-obj.params.dt/2 : obj.params.dt : max(dissociation_times)+obj.params.dt/2;
            [y, edges] = histcounts(dissociation_times, edges,'Normalization','pdf');

            % take the midpoint
            x = (edges(1:end-1)+edges(2:end))./2;

            % calculte one exponential fitting
            f_1 = @(A,tau_1,x)(A.*exp(-x./tau_1));
            f_2 = @(A,f1,tau_1,tau_2,x)(A.*(f1.*exp(-x./tau_1) + (1-f1).*exp(-x./tau_2)));
            f_3 = @(A,f1,f2,tau_1,tau_2,tau_3,x)(A.*(f1.*exp(-x./tau_1) + f2.*exp(-x./tau_2) + (1-f1-f2).*exp(-x./tau_3)));
            f_4 = @(A,f1,f2,f3,tau_1,tau_2,tau_3,tau_4,x)(A.*(f1.*exp(-x./tau_1) + f2.*exp(-x./tau_2) + f3.*exp(-x./tau_3) + (1-f1-f2-f3).*exp(-x./tau_4)));
            f_5 = @(A,f1,f2,f3,f4,tau_1,tau_2,tau_3,tau_4,tau_5,x)(A.*(f1.*exp(-x./tau_1) + f2.*exp(-x./tau_2) + f3.*exp(-x./tau_3) + f4.*exp(-x./tau_4) + (1-f1-f2-f3-f4).*exp(-x./tau_5)));

            f = {f_1, f_2, f_3, f_4, f_5};

            n_fit = min(5, floor(length(x)/2));
            fit_models = cell(n_fit,1);
            gof = cell(n_fit,1);
            bic = zeros(1,n_fit);

            for i = 1 : n_fit

                fo = fitoptions('Method','NonlinearLeastSquares',...
                    'Upper', [Inf ones(1,i-1) Inf(1,i)],...
                    'Lower', zeros(1,2*i),...
                    'StartPoint', ones(1,2*i),...
                    'Robust','LAR');

                [fit_models{i}, gof{i}] = fit(x', y', f{i}, fo);

                bic(i) = obj.bic_calculator(y, fit_models{i}(x)', 1);

                gof{i}.BIC = bic(i);
            end

            [~, best] = min(bic);

            fitModel = fit_models{best};
            fitStats = gof{best};

            % Plot Data and fits
            fig = figure("Name", "Dissociation Histogram");
            clf
            hold on

            x_grid = linspace(x(1),x(end),100);

            for i = 1 : best
                plot(x_grid,fit_models{i}(x_grid)','LineWidth',5, ...
                    "DisplayName", sprintf('#Pop=%d (BIC=%.2f)', i, bic(i)))
                if best == 1 && n_fit > 1
                    plot(x_grid,fit_models{2}(x_grid)',':','LineWidth',5,'Color',[0.5020 0.5020 0], ...
                        "DisplayName", sprintf('#Pop=2 (BIC=%.2f)', bic(2)))
                end
            end

            scatter(x,y,50,'k','Filled', "DisplayName", 'Empirical')

            legend("Location", "best")

            title('Dissociation times')
            xlabel('Time (seconds)')
            ylabel('pdf')

            axis([0 max(x) 0 max(y)])
            set(gca,'FontName','Arial','FontSize',12,'FontWeight','bold')
            shg


            if obj.params.exportFigures
                savefig(fig,fullfile(obj.params.figureFolderName,'DissociationHistogram.fig'))
            end


            fig = figure("Name", "Dissociation Fit BIC");
            clf
            hold on
            plot(1:n_fit, bic, 'LineWidth',5, 'Color', [0 0 0])
            scatter(best, bic(best), 100, 'red', 'filled')
            title('Dissociation fit BIC curve')
            xlabel('Number of exponentials')
            ylabel('BIC')
            set(gca, 'XTick', 1:n_fit)
            set(gca,'FontName','Arial','FontSize',12,'FontWeight','bold')
            shg

            if obj.params.exportFigures
                savefig(fig,fullfile(obj.params.figureFolderName,'DissociationBIC.fig'))
            end

            % Integrated Stan's codes --- End ---

            if length(y) < 3  % In case #observations<=#coefficients no CI
                disp("Warning: Data points too few for fitting -- Not fitting.")
                fitModel = [];
                return
            end

        end

        function ResultsGUI(obj)%
            % a gui to helo display results
            obj.handles.gui.mainFig   = figure('Name','Results','Units','Norm');
            obj.handles.gui.mainPanel = uipanel('Parent',obj.handles.gui.mainFig,'Units','norm');
            obj.handles.gui.checkbox.associationDissociation = ...
                uicontrol('style','checkbox',...
                'Parent',obj.handles.gui.mainFig,...
                'Units','norm',...
                'Position',[0.1,0.1, 0.2, 0.1],...
                'string','Association dissociation',...
                'Callback',@obj.PlotAssociationDissociationHistograms);

        end
    end

    methods (Static, Access=private)

        function [alpha,beta] = ComputeAlphaAndBetaMSD(times,meanSquareDisplacement)
            % compute the alpha (slope) and beta (intercept) of the best fit
            % (using linear regression) to the MSD input data
            % exclude the initial zero point to prevent divergence
            times              = [times(2)/50 times(2:end)];
            meanSquareDisplacement = [meanSquareDisplacement(2)/50 meanSquareDisplacement(2:end)];
            numPoints          = numel(meanSquareDisplacement);

            alpha = ((numPoints)*sum(log(meanSquareDisplacement).*log(times))-sum(log(times))*sum(log(meanSquareDisplacement)))/...
                ((numPoints)*sum(log(times).^2)-(sum(log(times))).^2);
            beta  = exp((1/(numPoints))*(sum(log(meanSquareDisplacement))-...
                alpha*sum(log(times))));

        end

        function msd          = ComputeMeanSquareDisplacement(traj)
            %               sd = (traj.x-traj.x(1)).^2+...
            %                    (traj.y-traj.y(1)).^2+...
            %                    (traj.z-traj.z(1)).^2;

            % Temp: compute msd along traj by successive dt jumps
            tData = [traj.x traj.y traj.z];  % work in µm
            N     = numel(traj.x);
            msd   = zeros(N,1);
            for tIdx = 0:(N-1)
                d   = [];
                next = 1;
                for mIdx = 1:N
                    if (mIdx+tIdx)<=N
                        d(next) = sum((tData(mIdx+tIdx,:)-tData(mIdx,:)).^2,2);
                        next    = next+1;
                    end
                end
                msd(tIdx+1) = sum(d)/(N-tIdx);%(next-1));%((next-1));
            end
            % gives the mean square displacement
        end

        function diffCoeff    = ComputeDiffusionCoefficient(traj,dt)
            diffCoeff = mean(sum(diff([traj.x traj.y traj.z],1).^2,2)./(2*3*dt),1);  % x y z in µm
        end

        function drift        = ComputeDrift(traj,dt)
            drift = mean(diff([traj.x traj.y traj.z],1)./dt,1);  % x y z in µm
        end

        function center       = ComputeCenter(traj)
            center = mean([traj.x, traj.y, traj.z]);
        end

        function dataOut      = Standardize(dataIn)
            dataOut = (dataIn-mean(dataIn))./std(dataIn);
        end

        function radius       = ComputeRadius(traj)
            radius = mean(sqrt((traj.x-traj.center(1)).^2+...
                (traj.y-traj.center(2)).^2+...
                (traj.z-traj.center(3)).^2));
        end

        function Lc           = ComputeLc(traj)
            % radius of confinement Lc
            Lc     = mean(std([traj.x traj.y traj.z]));
        end

        function Kc           = ComputeKc(traj,dt)
            % compute the constraint coefficient (spring constant)
            if all(traj.z==0)
                dimension =2;
            else
                dimension =3;
            end

            %               dimension = size(traj,2);
            N         = traj.numFrames; % number of steps in the trajectory
            kc        = zeros(1,dimension); % apparent spring constant
            pos       = [traj.x traj.y traj.z];
            for dIdx = 1:dimension
                x        = pos(1:N-1,dIdx)-traj.center(dIdx);
                y        = diff(pos(:,dIdx),1);
                r        = sortrows([x y]);
                kc(dIdx) = dot(r(:,1),r(:,2))/norm(r(:,1))^2;
            end
            %               Kc = -(mean(kc)/(traj.diffusionConst*dt*(N-1)));
            Kc = -(mean(kc)/(1*dt*(N-1)));
        end

        function traj         = NewTrajectoryDataStruct()
            % Create new trajectory data structure
            traj.ID                 = 0;  % trajecotry ID
            traj.numFrames          = 0;  % number of frames the trajectory appears in
            traj.frames             = []; % the frames the trajectory appears in
            traj.duration           = 0;  % time (sec)
            traj.x                  = 0;  % pos (nm)
            traj.y                  = 0;  % pos (nm)
            traj.z                  = 0;  % pos (nm)
            traj.alpha              = 0;  % moving alpha exponent in a window for MSD computation
            traj.beta               = 0;  % moving beta (bias) of moving window for MSD computation
            traj.msd                = {}; % sliding window square displacement
            traj.times              = {}; % sliding window times (s)
            traj.confined           = false; % indicator for confined point
            traj.confinementScore   = 0;  % score for confienment
            traj.associationTime    = []; % first association time
            traj.dissociationTime   = []; % first dissociation time
            traj.center             = [NaN,NaN,NaN]; % center of mass
            traj.radius             = NaN;% radius of a sphere confining the trajectory
            traj.mainAxis           = struct('a',0,'b',0,'x',[],'y',[],'z',[]); % linear fit to the main axis of the traj
            traj.spline             = cscvn(0); % spline structure
            traj.diffusionConst     = 0;  % apparent diffusion coefficient
            traj.drift              = [0,0,0]; % estimator for the drift
            traj.driftMagnitude     = 0;  % norm of the drift vectors along trajectory
            traj.Lc                 = 0;  % standard deviation from center of mass (confinent radius)
            traj.Kc                 = 0;  % apparent spring constant
            traj.angles             = []; % angles between successive trajecotry steps
            traj.displacement       = []; % successive displacement
            traj.originDataset      = {}; % name of the dataset (file) the trajectory belongs to
            traj.datasetIdx         = []; % index of the dataset the trajectory belongs to
        end

        function Trajectory2PDB(trajConf)
            % export trajectory X,Y,Z coordinates in pdb format

            FILE = fopen(trajConf.outFile, 'wt');

            %     #  output data
            for n = 1:length(trajConf.atomNum)
                %     # 1 -  6        Record name     "ATOM  "
                %     # 7 - 11        Integer         Atom serial number.
                %     # 13 - 16        Atom            Atom name.
                %     # 17             Character       Alternate location indicator.
                %     # 18 - 20        Residue name    Residue name.
                %     # 22             Character       Chain identifier.
                %     # 23 - 26        Integer         Residue sequence number.
                %     # 27             AChar           Code for insertion of residues.
                %     # 31 - 38        Real(8.3)       Orthogonal coordinates for X in Angstroms.
                %     # 39 - 46        Real(8.3)       Orthogonal coordinates for Y in Angstroms.
                %     # 47 - 54        Real(8.3)       Orthogonal coordinates for Z in Angstroms.
                %     # 55 - 60        Real(6.2)       Occupancy.
                %     # 61 - 66        Real(6.2)       Temperature factor (Default = 0.0).
                %     # 73 - 76        LString(4)      Segment identifier, left-justified.
                %     # 77 - 78        LString(2)      Element symbol, right-justified.
                %     # 79 - 80        LString(2)      Charge on the atom.

                fprintf(FILE,'%-6s%5u%3s%3.1s%3s %1.1s%4u%12.3f%8.3f%8.3f%6.2f%6.2f%12s%2s\n',...
                    trajConf.recordName{n}, trajConf.atomNum(n), trajConf.atomName{n},...
                    ' ',    trajConf.resName{n}, num2str(trajConf.chainID(n)), ...
                    trajConf.resNum(n),    trajConf.X(n)/100, trajConf.Y(n)/100, trajConf.Z(n)/100,...
                    trajConf.occupancy(n), trajConf.betaFactor(n), ...
                    trajConf.element{n},   num2str(trajConf.charge));

            end

            %      prepare a string with connectivity information for each monomer
            connectivity{1} = [1, 2];
            for cIdx = 2:length(trajConf.X)-1
                connectivity{cIdx} = [cIdx, cIdx-1,cIdx+1];
            end

            %         # COLUMNS         DATA TYPE        FIELD           DEFINITION
            %         # ---------------------------------------------------------------------------------
            %         #  1 -  6         Record name      "CONECT"
            %         #  7 - 11         Integer          serial          Atom serial number
            %         # 12 - 16         Integer          serial          Serial number of bonded atom
            %         # 17 - 21         Integer          serial          Serial number of bonded atom
            %         # 22 - 26         Integer          serial          Serial number of bonded atom
            %         # 27 - 31         Integer          serial          Serial number of bonded atom
            %         # 32 - 36         Integer          serial          Serial number of hydrogen bonded atom
            %         # 37 - 41         Integer          serial          Serial number of hydrogen bonded atom
            %         # 42 - 46         Integer          serial          Serial number of salt bridged atom
            %         # 47 - 51         Integer          serial          Serial number of hydrogen bonded atom
            %         # 52 - 56         Integer          serial          Serial number of hydrogen bonded atom
            %         # 57 - 61         Integer          serial          Serial number of salt bridged atom
            %
            %         # define the last entry

            connectivity{length(trajConf.X)} = [length(trajConf.X), length(trajConf.X)-1];

            for cIdx = 1:numel(connectivity)
                fprintf(FILE,'CONECT ');
                for sIdx = 1:length(connectivity{cIdx})
                    cm   = connectivity{cIdx}(sIdx);
                    fprintf(FILE,'%4.0d',cm);
                end
                fprintf(FILE,'\n');
            end

            fprintf( FILE, 'END\n');
            fclose(FILE);
        end

        function sp           = FitSplineToPath(x,y,z) % unused
            % fit a spline to a path data
            sp =[];% cscvn([x,y,z]');
        end

        function spCurvature  = SplineCurvature(sp,dt)
            % Compute the curvature of a spline curve
            % sp is a spline structure as the output of cscvn function
            nPts   = numel(sp.breaks);
            spVals = fnval(sp,linspace(sp.breaks(1), sp.breaks(end),nPts));
            spVals = [zeros(3,1), spVals, zeros(3,1)];% pad with zeros
            % compute forward first derivative
            dx     = zeros(nPts+1,1);
            dy     = zeros(nPts+1,1);
            dz     = zeros(nPts+1,1);
            for dIdx = 2:nPts+1
                dx(dIdx) = (spVals(1,dIdx+1)-spVals(1,dIdx))/dt;
                dy(dIdx) = (spVals(2,dIdx+1)-spVals(2,dIdx))/dt;
                dz(dIdx) = (spVals(3,dIdx+1)-spVals(3,dIdx))/dt;
            end
            % forward second derivative
            dx2 = zeros(nPts+1,1);
            dy2 = zeros(nPts+1,1);
            dz2 = zeros(nPts+1,1);
            for dIdx = 2:nPts+1
                dx2(dIdx) = (spVals(1,dIdx+1)-2*spVals(1,dIdx)+spVals(1,dIdx-1))/dt^2;
                dy2(dIdx) = (spVals(2,dIdx+1)-2*spVals(2,dIdx)+spVals(2,dIdx-1))/dt^2;
                dz2(dIdx) = (spVals(3,dIdx+1)-2*spVals(3,dIdx)+spVals(3,dIdx-1))/dt^2;
            end

            spCurvature = zeros(nPts+1,1);
            for dIdx = 1:nPts+1
                % unsigned curvature
                spCurvature(dIdx)= sqrt((dz2(dIdx)*dy(dIdx)-dy2(dIdx)*dz(dIdx)).^2 +...
                    (dx2(dIdx)*dz(dIdx)-dz2(dIdx)*dx(dIdx)).^2 +...
                    (dy2(dIdx)*dx(dIdx)- dx2(dIdx)*dy(dIdx).^2))/...
                    (dx(dIdx)^2+dy(dIdx)^2+dz(dIdx)^2).^(3/2);
                % signed curvature
                %                   spCurvature(dIdx)= ((dz2(dIdx)*dy(dIdx)-dy2(dIdx)*dz(dIdx)) +...
                %                                           (dx2(dIdx)*dz(dIdx)-dz2(dIdx)*dx(dIdx)) +...
                %                                           (dy2(dIdx)*dx(dIdx)- dx2(dIdx)*dy(dIdx)))/...
                %                                           (dx(dIdx)^2+dy(dIdx)^2+dz(dIdx)^2).^(3/2);

            end
            spCurvature =spCurvature(2:end);
        end

        function thetaVec = AnglesAlongTrajectory(traj)
            % compute angles between successive steps
            % trajIn in the form of numStepsXdimension
            numSteps = size(traj,1);
            thetaVec = zeros(numSteps-2,1);
            for stepIdx =1:numSteps-2
                v1 = traj(stepIdx+1,:)-traj(stepIdx,:);
                v2 = traj(stepIdx+2,:)-traj(stepIdx+1,:);
                thetaVec(stepIdx) = (dot(v1, v2)/...
                    (norm(v1)*norm(v2)));
            end
        end

        function displacement = ComputeDisplacement(x,y,z)
            displacement = sqrt(sum(diff([x,y,z]).^2,2))';
        end

        function bic = bic_calculator(y,y_fit,k)

            RSS = sum((y - y_fit).^2);
            n = numel(y);

            bic = n*log(RSS./n) + k*log(n);

        end

    end

end
