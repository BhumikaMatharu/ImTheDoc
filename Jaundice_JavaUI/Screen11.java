



import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.Socket;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;

class Screen11 {
	
	final JFrame f=new JFrame("Jaundice Detection");
	JFileChooser file=new JFileChooser();
	JLabel l,ans;

	Screen11()
	{
		
		f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		f.setLayout(null);
		
		l=new JLabel("                                                                                                                             Upload a file!");
		l.setBounds(0, 0, 850, 450);
		f.add(l);
		ans=new JLabel("Result");
		ans.setBounds(610, 500, 290, 140);
		f.add(ans);
		JButton upload=new JButton("Upload");
		upload.setBounds(50, 500, 90, 40);
		upload.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent arg0) {
				
				
			    file.setCurrentDirectory(new File(System.getProperty("user.home")));
			    FileNameExtensionFilter filter=new FileNameExtensionFilter("*.Images", "png","jpg");
			    file.addChoosableFileFilter(filter);
			    int result=file.showSaveDialog(null);
			    if(result==JFileChooser.APPROVE_OPTION)
			    {
			    	File selectedfile=file.getSelectedFile();
			    	String path=selectedfile.getAbsolutePath();
			    	OutputStream out=null;
			    	try{
			    		
			    	}
			    	catch(Exception e){
			    		e.printStackTrace();
			    	}
			    	l.setIcon(ResizeImage(path));
			    }
			    else if(result==JFileChooser.CANCEL_OPTION)
			    {
			    	System.out.println("No file select");
			    }
			}
		});
		f.add(upload);
		
		JButton clear=new JButton("Clear");
		clear.setBounds(330, 500, 90, 40);
		clear.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				
				l.setIcon(null);
				l.setText("                                                                                                                             Upload a file!");
				ans.setText("Result:");
			}
		});
		f.add(clear);
		
		JButton reload=new JButton("Reload");
		reload.setBounds(190, 500, 90, 40);
        reload.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent arg0) {
				
				if(l.getIcon()!=null)
				{
				JFileChooser file=new JFileChooser();
			    file.setCurrentDirectory(new File(System.getProperty("user.home")));
			    FileNameExtensionFilter filter=new FileNameExtensionFilter("*.Images", "png","jpg");
			    file.addChoosableFileFilter(filter);
			    int result=file.showSaveDialog(null);
			    if(result==JFileChooser.APPROVE_OPTION)
			    {
			    	File selectedfile=file.getSelectedFile();
			    	String path=selectedfile.getAbsolutePath();
			    	l.setIcon(ResizeImage(path));
			    }
			    else if(result==JFileChooser.CANCEL_OPTION)
			    {
			    	System.out.println("No file select");
			    }
			}
				else
				{
					l.setText(" Upload a file!");
				}
			}
			
		});
		
		f.add(reload);
		
		JButton run=new JButton("Run");
		run.setBounds(470, 500, 90, 40);
		run.addActionListener(new ActionListener() 
		{
			public void actionPerformed(ActionEvent arg0) {
				
				File selectedfile=file.getSelectedFile();
		    	String path=selectedfile.getAbsolutePath();
		    	Path pa=Paths.get(path);
		    	try{
		    		byte[] array = Files.readAllBytes(pa);
		    		byte[] encoded=Base64.getEncoder().encode(array);
		    		System.out.println(new String(encoded));
		    		String base64=new String(encoded);
		    		String url="http://127.0.0.1:5000/cloud1";
		    		System.out.println(url);
		    		URL url1 = new URL(url);
		    		HttpURLConnection uc=(HttpURLConnection)url1.openConnection();  
		            uc.setRequestMethod("POST");
		            uc.setDoOutput(true);
		            uc.setRequestProperty("Content-Type", "application/text"); 
		            uc.connect();
		            DataOutputStream dataOutputStream=new DataOutputStream(uc.getOutputStream());
		            dataOutputStream.writeBytes(base64);
		            dataOutputStream.flush();
		            dataOutputStream.close();
		            BufferedReader br = new BufferedReader(new InputStreamReader(uc.getInputStream()));      
		            StringBuilder sb=new StringBuilder();  
		            String line;
		            while ((line = br.readLine()) != null) {
		              sb.append(line);
		            }
		            String source=sb.toString();
		            System.out.println("dataaaaaaaaaaaaa "+source);
		            ans.setText(source);
		    		
		              
		    	}
		    	catch(Exception e){
		    		e.printStackTrace();
		    	}
		    	
			   
				
			}
			
			
		});
		
		f.add(run);
		
		
		f.setSize(850, 600);
	  f.setAlwaysOnTop(false);
		//f.setExtendedStateFrame.MAXIMIZED_BOTH);
	    f.setVisible(true);

	}
	
	public ImageIcon ResizeImage(String ImagePath)
	{
		ImageIcon myimage=new ImageIcon(ImagePath);
		Image img=myimage.getImage();
		Image newImg=img.getScaledInstance(l.getWidth(), l.getHeight(), Image.SCALE_SMOOTH);
		ImageIcon image=new ImageIcon(newImg);
		return image;
	}
	
	public static void main(String... args) {
		
		new Screen11();
		

	}

}

